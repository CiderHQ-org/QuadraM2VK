; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. Fifteen `air.*`
; texture symbols that the corpus reaches once or twice each and that no authored case covered.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_texture_variant_leftovers.bc'
source_filename = "validation/fixtures/public/kernel_texture_variant_leftovers.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_texture_variant_leftovers(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, ptr addrspace(1) %8, ptr addrspace(1) %9, ptr addrspace(1) %10, ptr addrspace(1) %11, ptr addrspace(1) %12, ptr addrspace(1) %13, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %14) local_unnamed_addr #0 {
  %16 = tail call ptr addrspace(2) @air.get_read_sampler() #6
  %17 = tail call { <4 x half>, i8 } @air.read_texture_2d_ms.v4f16(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %16, <2 x i32> <i32 1, i32 0>, i32 1, i32 1) #7
  %18 = extractvalue { <4 x half>, i8 } %17, 0
  %19 = bitcast <4 x half> %18 to <4 x i16>
  %20 = extractelement <4 x i16> %19, i64 0
  %21 = zext i16 %20 to i32
  store i32 %21, ptr addrspace(1) %14, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %22 = tail call { <4 x float>, i8 } @air.read_texture_2d_ms_array.v4f32(ptr addrspace(1) readonly captures(none) %1, ptr addrspace(2) %16, <2 x i32> zeroinitializer, i32 1, i32 2, i32 1) #7
  %23 = extractvalue { <4 x float>, i8 } %22, 0
  %24 = bitcast <4 x float> %23 to <4 x i32>
  %25 = extractelement <4 x i32> %24, i64 1
  %26 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 1
  store i32 %25, ptr addrspace(1) %26, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %27 = tail call { <4 x i16>, i8 } @air.read_texture_2d_ms_array.u.v4i16(ptr addrspace(1) readonly captures(none) %2, ptr addrspace(2) %16, <2 x i32> zeroinitializer, i32 1, i32 3, i32 1) #7
  %28 = extractvalue { <4 x i16>, i8 } %27, 0
  %29 = extractelement <4 x i16> %28, i64 2
  %30 = zext i16 %29 to i32
  %31 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 2
  store i32 %30, ptr addrspace(1) %31, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %32 = tail call { <4 x i16>, i8 } @air.read_texture_2d_array.u.v4i16(ptr addrspace(1) readonly captures(none) %3, ptr addrspace(2) %16, <2 x i32> splat (i32 1), i32 1, <2 x i32> zeroinitializer, i32 0, i32 0) #7
  %33 = extractvalue { <4 x i16>, i8 } %32, 0
  %34 = extractelement <4 x i16> %33, i64 3
  %35 = zext i16 %34 to i32
  %36 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 3
  store i32 %35, ptr addrspace(1) %36, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %37 = tail call { <4 x i32>, i8 } @air.gather_texture_2d.s.v4i32(ptr addrspace(1) readonly captures(none) %4, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> splat (float 5.625000e-01), i1 true, <2 x i32> zeroinitializer, i32 0, i32 0) #7
  %38 = extractvalue { <4 x i32>, i8 } %37, 0
  %39 = extractelement <4 x i32> %38, i64 0
  %40 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 4
  store i32 %39, ptr addrspace(1) %40, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %41 = tail call { <4 x i16>, i8 } @air.gather_texture_2d.s.v4i16(ptr addrspace(1) readonly captures(none) %5, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> splat (float 5.625000e-01), i1 true, <2 x i32> zeroinitializer, i32 0, i32 0) #7
  %42 = extractvalue { <4 x i16>, i8 } %41, 0
  %43 = extractelement <4 x i16> %42, i64 1
  %44 = zext i16 %43 to i32
  %45 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 5
  store i32 %44, ptr addrspace(1) %45, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %46 = tail call { <4 x i16>, i8 } @air.gather_texture_2d.u.v4i16(ptr addrspace(1) readonly captures(none) %6, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> splat (float 5.625000e-01), i1 true, <2 x i32> zeroinitializer, i32 0, i32 0) #7
  %47 = extractvalue { <4 x i16>, i8 } %46, 0
  %48 = extractelement <4 x i16> %47, i64 2
  %49 = zext i16 %48 to i32
  %50 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 6
  store i32 %49, ptr addrspace(1) %50, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %51 = tail call { <4 x i32>, i8 } @air.gather_texture_2d_array.s.v4i32(ptr addrspace(1) readonly captures(none) %7, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> splat (float 5.625000e-01), i32 0, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0) #7
  %52 = extractvalue { <4 x i32>, i8 } %51, 0
  %53 = extractelement <4 x i32> %52, i64 3
  %54 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 7
  store i32 %53, ptr addrspace(1) %54, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %55 = tail call { <4 x i32>, i8 } @air.sample_texture_3d.u.v4i32(ptr addrspace(1) readonly captures(none) %8, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 7.500000e-01, float 2.500000e-01, float 7.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #8
  %56 = extractvalue { <4 x i32>, i8 } %55, 0
  %57 = extractelement <4 x i32> %56, i64 0
  %58 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 8
  store i32 %57, ptr addrspace(1) %58, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %59 = tail call { <4 x i32>, i8 } @air.sample_texture_3d.s.v4i32(ptr addrspace(1) readonly captures(none) %9, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 7.500000e-01, float 2.500000e-01, float 7.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #8
  %60 = extractvalue { <4 x i32>, i8 } %59, 0
  %61 = extractelement <4 x i32> %60, i64 1
  %62 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 9
  store i32 %61, ptr addrspace(1) %62, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %63 = tail call { <4 x i16>, i8 } @air.sample_texture_2d.s.v4i16(ptr addrspace(1) readonly captures(none) %5, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> splat (float 5.625000e-01), i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #8
  %64 = extractvalue { <4 x i16>, i8 } %63, 0
  %65 = extractelement <4 x i16> %64, i64 2
  %66 = zext i16 %65 to i32
  %67 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 10
  store i32 %66, ptr addrspace(1) %67, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %68 = tail call { <4 x half>, i8 } @air.read_texture_buffer_1d.v4f16(ptr addrspace(1) readonly captures(none) %10, ptr addrspace(2) %16, i32 2, i32 1) #7
  %69 = extractvalue { <4 x half>, i8 } %68, 0
  %70 = bitcast <4 x half> %69 to <4 x i16>
  %71 = extractelement <4 x i16> %70, i64 0
  %72 = zext i16 %71 to i32
  %73 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 11
  store i32 %72, ptr addrspace(1) %73, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  tail call void @air.write_texture_buffer_1d.v4f16(ptr addrspace(1) captures(none) %11, i32 1, <4 x half> <half 0xH3800, half 0xHBE00, half 0xH4800, half 0xH3400>, i32 2) #9, !alias.scope !42, !noalias !39
  tail call void @air.write_texture_2d_array.i16.s.v4i32(ptr addrspace(1) captures(none) %12, <2 x i16> <i16 1, i16 0>, i16 0, <4 x i32> <i32 -7, i32 11, i32 0, i32 2147483647>, i16 0, i32 2) #9, !alias.scope !42, !noalias !39
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none) %13, <2 x i32> zeroinitializer, <4 x float> <float 1.500000e+00, float -2.250000e+00, float 0.000000e+00, float 6.400000e+01>, i32 0, i32 3) #9, !alias.scope !42, !noalias !39
  tail call void @air.fence_texture_2d(ptr addrspace(1) captures(none) %13) #10
  %74 = tail call ptr addrspace(2) @air.get_read_sampler() #6
  %75 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %13, ptr addrspace(2) %74, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 3) #7
  %76 = extractvalue { <4 x float>, i8 } %75, 0
  %77 = bitcast <4 x float> %76 to <4 x i32>
  %78 = extractelement <4 x i32> %77, i64 0
  %79 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 12
  store i32 %78, ptr addrspace(1) %79, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  %80 = extractelement <4 x i32> %77, i64 3
  %81 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 13
  store i32 %80, ptr addrspace(1) %81, align 4, !tbaa !35, !alias.scope !39, !noalias !42
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: read)
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x half>, i8 } @air.read_texture_2d_ms.v4f16(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.read_texture_2d_ms_array.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i16>, i8 } @air.read_texture_2d_ms_array.u.v4i16(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i16>, i8 } @air.read_texture_2d_array.u.v4i16(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i32>, i8 } @air.gather_texture_2d.s.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i16>, i8 } @air.gather_texture_2d.s.v4i16(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i16>, i8 } @air.gather_texture_2d.u.v4i16(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i32>, i8 } @air.gather_texture_2d_array.s.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i32, i1, <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i32>, i8 } @air.sample_texture_3d.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <3 x float>, i1, <3 x i32>, i1, float, float, i32) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i32>, i8 } @air.sample_texture_3d.s.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <3 x float>, i1, <3 x i32>, i1, float, float, i32) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i16>, i8 } @air.sample_texture_2d.s.v4i16(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x half>, i8 } @air.read_texture_buffer_1d.v4f16(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_buffer_1d.v4f16(ptr addrspace(1) captures(none), i32, <4 x half>, i32) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_2d_array.i16.s.v4i32(ptr addrspace(1) captures(none), <2 x i16>, i16, <4 x i32>, i16, i32) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none), <2 x i32>, <4 x float>, i32, i32) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind willreturn
declare void @air.fence_texture_2d(ptr addrspace(1) captures(none)) local_unnamed_addr #5

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #2

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { convergent mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nounwind willreturn }
attributes #6 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #7 = { nounwind willreturn memory(argmem: read) }
attributes #8 = { convergent nounwind willreturn memory(argmem: read) }
attributes #9 = { nounwind willreturn memory(argmem: readwrite) }
attributes #10 = { nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!27, !28, !29}
!air.sampler_states = !{!30}
!llvm.ident = !{!31}
!air.version = !{!32}
!air.language_version = !{!33}
!air.source_file_name = !{!34}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_texture_variant_leftovers, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<half, read>", !"air.arg_name", !"msh"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms_array<float, read>", !"air.arg_name", !"msaf"}
!14 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms_array<ushort, read>", !"air.arg_name", !"msau"}
!15 = !{i32 3, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<ushort, sample>", !"air.arg_name", !"arru"}
!16 = !{i32 4, !"air.texture", !"air.location_index", i32 4, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<int, sample>", !"air.arg_name", !"ti"}
!17 = !{i32 5, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<short, sample>", !"air.arg_name", !"ts"}
!18 = !{i32 6, !"air.texture", !"air.location_index", i32 6, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<ushort, sample>", !"air.arg_name", !"tu"}
!19 = !{i32 7, !"air.texture", !"air.location_index", i32 7, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<int, sample>", !"air.arg_name", !"tai"}
!20 = !{i32 8, !"air.texture", !"air.location_index", i32 8, i32 1, !"air.sample", !"air.arg_type_name", !"texture3d<uint, sample>", !"air.arg_name", !"t3u"}
!21 = !{i32 9, !"air.texture", !"air.location_index", i32 9, i32 1, !"air.sample", !"air.arg_type_name", !"texture3d<int, sample>", !"air.arg_name", !"t3i"}
!22 = !{i32 10, !"air.texture", !"air.location_index", i32 10, i32 1, !"air.read", !"air.arg_type_name", !"texture_buffer<half, read>", !"air.arg_name", !"tbr"}
!23 = !{i32 11, !"air.texture", !"air.location_index", i32 11, i32 1, !"air.write", !"air.arg_type_name", !"texture_buffer<half, write>", !"air.arg_name", !"tbw"}
!24 = !{i32 12, !"air.texture", !"air.location_index", i32 12, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<int, write>", !"air.arg_name", !"taw"}
!25 = !{i32 13, !"air.texture", !"air.location_index", i32 13, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"trw"}
!26 = !{i32 14, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!27 = !{!"air.compile.denorms_disable"}
!28 = !{!"air.compile.fast_math_enable"}
!29 = !{!"air.compile.framebuffer_fetch_enable"}
!30 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!31 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!32 = !{i32 2, i32 8, i32 0}
!33 = !{!"Metal", i32 4, i32 0, i32 0}
!34 = !{!"/private/tmp/tex/k.metal"}
!35 = !{!36, !36, i64 0}
!36 = !{!"int", !37, i64 0}
!37 = !{!"omnipotent char", !38, i64 0}
!38 = !{!"Simple C++ TBAA"}
!39 = !{!40}
!40 = distinct !{!40, !41, !"air-alias-scope-arg(14)"}
!41 = distinct !{!41, !"air-alias-scopes(kernel_texture_variant_leftovers)"}
!42 = !{!43}
!43 = distinct !{!43, !41, !"air-alias-scope-textures"}
