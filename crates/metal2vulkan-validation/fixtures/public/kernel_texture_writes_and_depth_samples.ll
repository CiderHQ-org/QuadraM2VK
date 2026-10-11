; ModuleID = '/tmp/mkfix2/w3.bc'
source_filename = "validation/fixtures/public/kernel_texture_writes_and_depth_samples.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_texture_writes_and_depth_samples(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(2) readonly captures(none) %6, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %7, i32 noundef %8) local_unnamed_addr #0 {
  %10 = and i32 %8, 1
  %11 = trunc i32 %10 to i16
  %12 = lshr i32 %8, 1
  %13 = trunc i32 %12 to i16
  %14 = trunc i32 %8 to i16
  %15 = add i16 %14, 100
  %16 = insertelement <4 x i16> undef, i16 %15, i64 0
  %17 = add i16 %14, 200
  %18 = insertelement <4 x i16> %16, i16 %17, i64 1
  %19 = add i16 %14, 300
  %20 = insertelement <4 x i16> %18, i16 %19, i64 2
  %21 = add i16 %14, 400
  %22 = insertelement <4 x i16> %20, i16 %21, i64 3
  %23 = insertelement <3 x i16> undef, i16 %11, i64 0
  %24 = insertelement <3 x i16> %23, i16 %13, i64 1
  %25 = insertelement <3 x i16> %24, i16 %11, i64 2
  tail call void @air.write_texture_3d.i16.u.v4i16(ptr addrspace(1) captures(none) %0, <3 x i16> %25, <4 x i16> %22, i16 0, i32 2) #5, !alias.scope !28, !noalias !31
  %26 = tail call fast float @air.convert.f.f32.u.i32(i32 %8) #6
  %27 = fadd fast float %26, 5.000000e-01
  %28 = insertelement <4 x float> undef, float %27, i64 0
  %29 = fadd fast float %26, 1.500000e+00
  %30 = insertelement <4 x float> %28, float %29, i64 1
  %31 = fadd fast float %26, 2.500000e+00
  %32 = insertelement <4 x float> %30, float %31, i64 2
  %33 = fadd fast float %26, 3.500000e+00
  %34 = insertelement <4 x float> %32, float %33, i64 3
  tail call void @air.write_texture_1d.i16.v4f32(ptr addrspace(1) captures(none) %1, i16 %14, <4 x float> %34, i16 0, i32 2) #5, !alias.scope !28, !noalias !31
  %35 = tail call fast half @air.convert.f.f16.u.i32(i32 %8) #6
  %36 = fadd fast half %35, 0xH3800
  %37 = insertelement <4 x half> undef, half %36, i64 0
  %38 = fadd fast half %35, 0xH3E00
  %39 = insertelement <4 x half> %37, half %38, i64 1
  %40 = fadd fast half %35, 0xH4100
  %41 = insertelement <4 x half> %39, half %40, i64 2
  %42 = fadd fast half %35, 0xH4300
  %43 = insertelement <4 x half> %41, half %42, i64 3
  %44 = insertelement <3 x i16> undef, i16 %13, i64 0
  %45 = insertelement <3 x i16> %44, i16 %11, i64 1
  %46 = insertelement <3 x i16> %45, i16 %13, i64 2
  tail call void @air.write_texture_3d.i16.v4f16(ptr addrspace(1) captures(none) %2, <3 x i16> %46, <4 x half> %43, i16 0, i32 2) #5, !alias.scope !28, !noalias !31
  %47 = fmul fast float %26, 2.500000e-01
  %48 = fadd fast float %47, 1.250000e-01
  %49 = tail call { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none) %3, ptr addrspace(2) readonly captures(none) %6, float %48, i32 %10, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #7, !alias.scope !34, !noalias !35
  switch i32 %8, label %50 [
    i32 0, label %54
    i32 1, label %53
  ]

50:                                               ; preds = %9
  %51 = icmp eq i32 %8, 2
  %52 = select fast i1 %51, <3 x float> <float 5.000000e-01, float 1.000000e+00, float 5.000000e-01>, <3 x float> <float 5.000000e-01, float -1.000000e+00, float 5.000000e-01>
  br label %54

53:                                               ; preds = %9
  br label %54

54:                                               ; preds = %53, %50, %9
  %55 = phi fast <3 x float> [ <float 1.000000e+00, float 5.000000e-01, float 5.000000e-01>, %9 ], [ %52, %50 ], [ <float -1.000000e+00, float 5.000000e-01, float 5.000000e-01>, %53 ]
  %56 = extractvalue { <4 x float>, i8 } %49, 0
  %57 = tail call { float, i8 } @air.sample_depth_cube.f32(ptr addrspace(1) readonly captures(none) %4, ptr addrspace(2) readonly captures(none) %6, i32 1, <3 x float> %55, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #7, !alias.scope !34, !noalias !35
  %58 = extractvalue { float, i8 } %57, 0
  %59 = shl i32 %8, 2
  %60 = bitcast <4 x float> %56 to <4 x i32>
  %61 = extractelement <4 x i32> %60, i64 0
  %62 = zext i32 %59 to i64
  %63 = getelementptr inbounds i32, ptr addrspace(1) %7, i64 %62
  store i32 %61, ptr addrspace(1) %63, align 4, !tbaa !36, !alias.scope !35, !noalias !34
  %64 = or i32 %59, 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i32, ptr addrspace(1) %7, i64 %65
  %67 = bitcast ptr addrspace(1) %66 to ptr addrspace(1)
  store float %58, ptr addrspace(1) %67, align 4, !tbaa !36, !alias.scope !35, !noalias !34
  %68 = tail call i32 @air.get_num_mip_levels_depth_cube(ptr addrspace(1) readonly captures(none) %4) #8, !alias.scope !28, !noalias !31
  %69 = or i32 %59, 2
  %70 = zext i32 %69 to i64
  %71 = getelementptr inbounds i32, ptr addrspace(1) %7, i64 %70
  store i32 %68, ptr addrspace(1) %71, align 4, !tbaa !36, !alias.scope !35, !noalias !34
  %72 = tail call i32 @air.get_num_mip_levels_depth_2d(ptr addrspace(1) readonly captures(none) %5) #8, !alias.scope !28, !noalias !31
  %73 = or i32 %59, 3
  %74 = zext i32 %73 to i64
  %75 = getelementptr inbounds i32, ptr addrspace(1) %7, i64 %74
  store i32 %72, ptr addrspace(1) %75, align 4, !tbaa !36, !alias.scope !35, !noalias !34
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_3d.i16.u.v4i16(ptr addrspace(1) captures(none), <3 x i16>, <4 x i16>, i16, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_1d.i16.v4f32(ptr addrspace(1) captures(none), i16, <4 x float>, i16, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_3d.i16.v4f16(ptr addrspace(1) captures(none), <3 x i16>, <4 x half>, i16, i32) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), float, i32, i1, i32, i1, float, float, i32) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { float, i8 } @air.sample_depth_cube.f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), i32, <3 x float>, i1, float, float, i32) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_num_mip_levels_depth_cube(ptr addrspace(1) readonly captures(none)) local_unnamed_addr #4

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_num_mip_levels_depth_2d(ptr addrspace(1) readonly captures(none)) local_unnamed_addr #4

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { convergent mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #5 = { nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { nounwind willreturn memory(none) }
attributes #7 = { convergent nounwind willreturn memory(argmem: read) }
attributes #8 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!21, !22, !23}
!llvm.ident = !{!24}
!air.version = !{!25}
!air.language_version = !{!26}
!air.source_file_name = !{!27}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_texture_writes_and_depth_samples, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19, !20}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture3d<ushort, write>", !"air.arg_name", !"vol"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture1d<float, write>", !"air.arg_name", !"line"}
!14 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture3d<half, write>", !"air.arg_name", !"hvol"}
!15 = !{i32 3, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.sample", !"air.arg_type_name", !"texture1d_array<float, sample>", !"air.arg_name", !"larr"}
!16 = !{i32 4, !"air.texture", !"air.location_index", i32 4, i32 1, !"air.sample", !"air.arg_type_name", !"depthcube<float, sample>", !"air.arg_name", !"dcube"}
!17 = !{i32 5, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"d2"}
!18 = !{i32 6, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"samp"}
!19 = !{i32 7, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!20 = !{i32 8, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!21 = !{!"air.compile.denorms_disable"}
!22 = !{!"air.compile.fast_math_enable"}
!23 = !{!"air.compile.framebuffer_fetch_enable"}
!24 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!25 = !{i32 2, i32 8, i32 0}
!26 = !{!"Metal", i32 4, i32 0, i32 0}
!27 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_texture_writes_and_depth_samples.metal"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-textures"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_texture_writes_and_depth_samples)"}
!31 = !{!32, !33}
!32 = distinct !{!32, !30, !"air-alias-scope-samplers"}
!33 = distinct !{!33, !30, !"air-alias-scope-arg(7)"}
!34 = !{!29, !32}
!35 = !{!33}
!36 = !{!37, !37, i64 0}
!37 = !{!"int", !38, i64 0}
!38 = !{!"omnipotent char", !39, i64 0}
!39 = !{!"Simple C++ TBAA"}
