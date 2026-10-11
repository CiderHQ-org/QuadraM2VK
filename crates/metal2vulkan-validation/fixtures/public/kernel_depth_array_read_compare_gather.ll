; ModuleID = '/tmp/da/f.bc'
source_filename = "validation/fixtures/public/kernel_depth_array_read_compare_gather.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797600559177, i64 0], align 8
@__air_sampler_state.1 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

; Function Attrs: convergent mustprogress nofree nounwind willreturn
define void @kernel_depth_array_read_compare_gather(ptr addrspace(1) %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = trunc i32 %2 to i16
  %5 = and i16 %4, 3
  %6 = lshr i32 %2, 2
  %7 = trunc i32 %6 to i16
  %8 = insertelement <2 x i16> undef, i16 %5, i64 0
  %9 = insertelement <2 x i16> %8, i16 %7, i64 1
  %10 = tail call ptr addrspace(2) @air.get_read_sampler() #5
  %11 = tail call { float, i8 } @air.read_depth_2d_array.i16.f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, i32 1, <2 x i16> %9, i16 0, <2 x i16> zeroinitializer, i16 0, i32 0) #6
  %12 = extractvalue { float, i8 } %11, 0
  %13 = tail call { float, i8 } @air.read_depth_2d_array.i16.f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %10, i32 1, <2 x i16> %9, i16 1, <2 x i16> zeroinitializer, i16 0, i32 0) #6
  %14 = extractvalue { float, i8 } %13, 0
  %15 = tail call fast float @air.convert.f.f32.u.i16(i16 %5) #7
  %16 = fadd fast float %15, 5.000000e-01
  %17 = fmul fast float %16, 2.500000e-01
  %18 = insertelement <2 x float> undef, float %17, i64 0
  %19 = tail call fast float @air.convert.f.f32.u.i16(i16 %7) #7
  %20 = fadd fast float %19, 5.000000e-01
  %21 = fmul fast float %20, 5.000000e-01
  %22 = insertelement <2 x float> %18, float %21, i64 1
  %23 = mul i32 %2, 3
  %24 = and i32 %23, 7
  %25 = tail call fast float @air.convert.f.f32.u.i32(i32 %24) #7
  %26 = fmul fast float %25, 6.250000e-02
  %27 = fadd fast float %26, 3.125000e-02
  %28 = tail call { float, i8 } @air.sample_compare_depth_2d_array.f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) readonly captures(none) @__air_sampler_state, i32 1, <2 x float> %22, i32 0, float %27, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #8
  %29 = extractvalue { float, i8 } %28, 0
  %30 = fadd fast float %26, 5.312500e-01
  %31 = tail call { float, i8 } @air.sample_compare_depth_2d_array.f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) readonly captures(none) @__air_sampler_state, i32 1, <2 x float> %22, i32 1, float %30, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #8
  %32 = extractvalue { float, i8 } %31, 0
  %33 = urem i32 %2, 3
  %34 = tail call fast float @air.convert.f.f32.u.i32(i32 %33) #7
  %35 = fmul fast float %34, 2.500000e-01
  %36 = fadd fast float %35, 2.500000e-01
  %37 = insertelement <2 x float> <float poison, float 5.000000e-01>, float %36, i64 0
  %38 = and i32 %2, 1
  %39 = tail call { <4 x float>, i8 } @air.gather_depth_2d_array.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) readonly captures(none) @__air_sampler_state.1, i32 1, <2 x float> %37, i32 %38, i1 true, <2 x i32> zeroinitializer, i32 0) #6
  %40 = extractvalue { <4 x float>, i8 } %39, 0
  %41 = mul i32 %2, 10
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %42
  %44 = bitcast ptr addrspace(1) %43 to ptr addrspace(1)
  store float %12, ptr addrspace(1) %44, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %45 = or i32 %41, 1
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %46
  %48 = bitcast ptr addrspace(1) %47 to ptr addrspace(1)
  store float %14, ptr addrspace(1) %48, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %49 = add i32 %41, 2
  %50 = zext i32 %49 to i64
  %51 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %50
  %52 = bitcast ptr addrspace(1) %51 to ptr addrspace(1)
  store float %29, ptr addrspace(1) %52, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %53 = add i32 %41, 3
  %54 = zext i32 %53 to i64
  %55 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %54
  %56 = bitcast ptr addrspace(1) %55 to ptr addrspace(1)
  store float %32, ptr addrspace(1) %56, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %57 = bitcast <4 x float> %40 to <4 x i32>
  %58 = extractelement <4 x i32> %57, i64 0
  %59 = add i32 %41, 4
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %60
  store i32 %58, ptr addrspace(1) %61, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %62 = extractelement <4 x i32> %57, i64 1
  %63 = add i32 %41, 5
  %64 = zext i32 %63 to i64
  %65 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %64
  store i32 %62, ptr addrspace(1) %65, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %66 = extractelement <4 x i32> %57, i64 2
  %67 = add i32 %41, 6
  %68 = zext i32 %67 to i64
  %69 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %68
  store i32 %66, ptr addrspace(1) %69, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %70 = extractelement <4 x i32> %57, i64 3
  %71 = add i32 %41, 7
  %72 = zext i32 %71 to i64
  %73 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %72
  store i32 %70, ptr addrspace(1) %73, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %74 = tail call i32 @air.get_width_depth_2d_array(ptr addrspace(1) readonly captures(none) %0, i32 0) #6, !alias.scope !31, !noalias !28
  %75 = add i32 %41, 8
  %76 = zext i32 %75 to i64
  %77 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %76
  store i32 %74, ptr addrspace(1) %77, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %78 = tail call i32 @air.get_height_depth_2d_array(ptr addrspace(1) readonly captures(none) %0, i32 0) #6, !alias.scope !31, !noalias !28
  %79 = add i32 %41, 9
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %80
  store i32 %78, ptr addrspace(1) %81, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i16(i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: read)
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { float, i8 } @air.read_depth_2d_array.i16.f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, <2 x i16>, i16, <2 x i16>, i16, i32) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { float, i8 } @air.sample_compare_depth_2d_array.f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), i32, <2 x float>, i32, float, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr #4

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.gather_depth_2d_array.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), i32, <2 x float>, i32, i1, <2 x i32>, i32) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_width_depth_2d_array(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_height_depth_2d_array(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #3

attributes #0 = { convergent mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #3 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { convergent mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #5 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #6 = { nounwind willreturn memory(argmem: read) }
attributes #7 = { nounwind willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!air.sampler_states = !{!18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}
!air.source_file_name = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_depth_array_read_compare_gather, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d_array<float, sample>", !"air.arg_name", !"darr"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!19 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.1}
!20 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 4, i32 0, i32 0}
!23 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_depth_array_read_compare_gather.metal"}
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(1)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_depth_array_read_compare_gather)"}
!31 = !{!32}
!32 = distinct !{!32, !30, !"air-alias-scope-textures"}
