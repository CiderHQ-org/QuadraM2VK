; ModuleID = '/tmp/mkfix2/t2.bc'
source_filename = "validation/fixtures/public/kernel_texture_reads_and_queries.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nounwind willreturn
define void @kernel_texture_reads_and_queries(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) readonly captures(none) %4, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %5, i32 noundef %6) local_unnamed_addr #0 {
  %8 = and i32 %6, 1
  %9 = lshr i32 %6, 1
  %10 = insertelement <2 x i32> undef, i32 %8, i64 0
  %11 = insertelement <2 x i32> %10, i32 %9, i64 1
  %12 = tail call ptr addrspace(2) @air.get_read_sampler() #3
  %13 = tail call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %12, <2 x i32> %11, i32 %6, i32 0, i32 1) #4
  %14 = extractvalue { <4 x float>, i8 } %13, 0
  %15 = tail call { float, i8 } @air.read_depth_2d_array.f32(ptr addrspace(1) readonly captures(none) %1, ptr addrspace(2) %12, i32 1, <2 x i32> %11, i32 %8, <2 x i32> zeroinitializer, i32 0, i32 1) #4
  %16 = extractvalue { float, i8 } %15, 0
  %17 = trunc i32 %8 to i16
  %18 = insertelement <2 x i16> undef, i16 %17, i64 0
  %19 = trunc i32 %9 to i16
  %20 = insertelement <2 x i16> %18, i16 %19, i64 1
  %21 = tail call { <4 x i16>, i8 } @air.read_texture_2d_array.i16.u.v4i16(ptr addrspace(1) readonly captures(none) %2, ptr addrspace(2) %12, <2 x i16> %20, i16 %19, <2 x i16> zeroinitializer, i16 0, i32 1) #4
  %22 = extractvalue { <4 x i16>, i8 } %21, 0
  %23 = tail call i32 @air.get_width_texture_1d(ptr addrspace(1) readonly captures(none) %3, i32 0) #4, !alias.scope !26, !noalias !29
  %24 = tail call i1 @air.is_null_texture_1d(ptr addrspace(1) readonly captures(none) %3) #4, !alias.scope !26, !noalias !29
  %25 = tail call i1 @air.is_null_texture_3d(ptr addrspace(1) readonly captures(none) %4) #4, !alias.scope !26, !noalias !29
  %26 = mul i32 %6, 7
  %27 = bitcast <4 x float> %14 to <4 x i32>
  %28 = extractelement <4 x i32> %27, i64 0
  %29 = zext i32 %26 to i64
  %30 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %29
  store i32 %28, ptr addrspace(1) %30, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %31 = extractelement <4 x i32> %27, i64 3
  %32 = add i32 %26, 1
  %33 = zext i32 %32 to i64
  %34 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %33
  store i32 %31, ptr addrspace(1) %34, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %35 = add i32 %26, 2
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %36
  %38 = bitcast ptr addrspace(1) %37 to ptr addrspace(1)
  store float %16, ptr addrspace(1) %38, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %39 = extractelement <4 x i16> %22, i64 0
  %40 = zext i16 %39 to i32
  %41 = add i32 %26, 3
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %42
  store i32 %40, ptr addrspace(1) %43, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %44 = extractelement <4 x i16> %22, i64 3
  %45 = zext i16 %44 to i32
  %46 = add i32 %26, 4
  %47 = zext i32 %46 to i64
  %48 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %47
  store i32 %45, ptr addrspace(1) %48, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %49 = add i32 %26, 5
  %50 = zext i32 %49 to i64
  %51 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %50
  store i32 %23, ptr addrspace(1) %51, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %52 = select i1 %24, i32 2, i32 0
  %53 = zext i1 %25 to i32
  %54 = or i32 %52, %53
  %55 = add i32 %26, 6
  %56 = zext i32 %55 to i64
  %57 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %56
  store i32 %54, ptr addrspace(1) %57, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: read)
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { float, i8 } @air.read_depth_2d_array.f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, <2 x i32>, i32, <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i16>, i8 } @air.read_texture_2d_array.i16.u.v4i16(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i16>, i16, <2 x i16>, i16, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_width_texture_1d(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i1 @air.is_null_texture_1d(ptr addrspace(1) readonly captures(none)) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i1 @air.is_null_texture_3d(ptr addrspace(1) readonly captures(none)) local_unnamed_addr #2

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #4 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!19, !20, !21}
!llvm.ident = !{!22}
!air.version = !{!23}
!air.language_version = !{!24}
!air.source_file_name = !{!25}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_texture_reads_and_queries, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texturecube<float, read>", !"air.arg_name", !"cube"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"depth2d_array<float, read>", !"air.arg_name", !"darr"}
!14 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_array<ushort, read>", !"air.arg_name", !"uarr"}
!15 = !{i32 3, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.read", !"air.arg_type_name", !"texture1d<float, read>", !"air.arg_name", !"line"}
!16 = !{i32 4, !"air.texture", !"air.location_index", i32 4, i32 1, !"air.read", !"air.arg_type_name", !"texture3d<float, read>", !"air.arg_name", !"vol"}
!17 = !{i32 5, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!18 = !{i32 6, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 4, i32 0, i32 0}
!25 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_texture_reads_and_queries.metal"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-textures"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_texture_reads_and_queries)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(5)"}
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C++ TBAA"}
