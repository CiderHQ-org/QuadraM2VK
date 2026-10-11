; ModuleID = 'k.fix.ll'
source_filename = "kernel_texture_select_queries.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

; Function Attrs: convergent mustprogress nofree nounwind willreturn
define void @texture_select_queries(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(2) readonly align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %3, i32 %4) local_unnamed_addr #0 {
  %6 = and i32 %4, 1
  %7 = load i32, ptr addrspace(2) %3, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %8 = icmp eq i32 %6, %7
  %9 = select i1 %8, ptr addrspace(1) %0, ptr addrspace(1) %1
  %10 = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none) %9, i32 0) #5
  %11 = tail call fast float @air.convert.f.f32.u.i32(i32 %10) #6
  %12 = shl i32 %4, 2
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %13
  store float %11, ptr addrspace(1) %14, align 4, !tbaa !35, !alias.scope !37, !noalias !38
  %15 = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) readonly captures(none) %9, i32 0) #5
  %16 = tail call fast float @air.convert.f.f32.u.i32(i32 %15) #6
  %17 = or i32 %12, 1
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %18
  store float %16, ptr addrspace(1) %19, align 4, !tbaa !35, !alias.scope !37, !noalias !38
  %20 = tail call ptr addrspace(2) @air.get_read_sampler() #7
  %21 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %9, ptr addrspace(2) %20, <2 x i32> <i32 1, i32 0>, <2 x i32> zeroinitializer, i32 0, i32 0) #5
  %22 = extractvalue { <4 x float>, i8 } %21, 0
  %23 = extractelement <4 x float> %22, i64 0
  %24 = or i32 %12, 2
  %25 = zext i32 %24 to i64
  %26 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %25
  store float %23, ptr addrspace(1) %26, align 4, !tbaa !35, !alias.scope !37, !noalias !38
  %27 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %9, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 0x3FE99999A0000000, float 0x3FD3333340000000>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #8
  %28 = extractvalue { <4 x float>, i8 } %27, 0
  %29 = extractelement <4 x float> %28, i64 0
  %30 = or i32 %12, 3
  %31 = zext i32 %30 to i64
  %32 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %31
  store float %29, ptr addrspace(1) %32, align 4, !tbaa !35, !alias.scope !37, !noalias !38
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_height_texture_2d(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: read)
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr #4

attributes #0 = { convergent mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #4 = { convergent mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #5 = { nounwind willreturn memory(argmem: read) }
attributes #6 = { nounwind willreturn memory(none) }
attributes #7 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #8 = { convergent nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!air.sampler_states = !{!20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @texture_select_queries, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"a"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"b"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"pick"}
!16 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 3, i32 0, i32 0}
!24 = !{!"/private/tmp/texsel/kernel_texture_select_queries.metal"}
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C++ TBAA"}
!29 = !{!30}
!30 = distinct !{!30, !31, !"air-alias-scope-arg(3)"}
!31 = distinct !{!31, !"air-alias-scopes(texture_select_queries)"}
!32 = !{!33, !34}
!33 = distinct !{!33, !31, !"air-alias-scope-textures"}
!34 = distinct !{!34, !31, !"air-alias-scope-arg(2)"}
!35 = !{!36, !36, i64 0}
!36 = !{!"float", !27, i64 0}
!37 = !{!34}
!38 = !{!33, !30}
