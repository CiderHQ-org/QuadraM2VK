; ModuleID = '/tmp/da/sc.bc'
source_filename = "validation/fixtures/public/kernel_imageblock_scaled_tile.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_imageblock_scaled_tile(ptr addrspace(1) %0, ptr addrspace(1) %1, %"struct.metal::_imageblock_base" %2, <2 x i16> noundef %3, <2 x i16> noundef %4) local_unnamed_addr #0 {
  %6 = shl <2 x i16> %3, splat (i16 1)
  %7 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %6, i32 0, i16 0) #5
  %8 = extractelement <2 x i16> %6, i64 0
  %9 = tail call fast float @air.convert.f.f32.u.i16(i16 %8) #6
  %10 = insertelement <4 x float> undef, float %9, i64 0
  %11 = extractelement <2 x i16> %6, i64 1
  %12 = tail call fast float @air.convert.f.f32.u.i16(i16 %11) #6
  %13 = insertelement <4 x float> %10, float %12, i64 1
  %14 = fmul fast float %9, 5.000000e-01
  %15 = insertelement <4 x float> %13, float %14, i64 2
  %16 = fmul fast float %12, 2.500000e-01
  %17 = insertelement <4 x float> %15, float %16, i64 3
  %18 = bitcast ptr addrspace(4) %7 to ptr addrspace(4)
  store <4 x float> %17, ptr addrspace(4) %18, align 16, !tbaa !25
  %19 = fmul fast float %12, 1.600000e+01
  %20 = fadd fast float %19, %9
  %21 = getelementptr inbounds i8, ptr addrspace(4) %7, i64 16
  %22 = bitcast ptr addrspace(4) %21 to ptr addrspace(4)
  store float %20, ptr addrspace(4) %22, align 16, !tbaa !28
  %23 = or <2 x i16> %6, <i16 1, i16 0>
  %24 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %23, i32 0, i16 0) #5
  %25 = extractelement <2 x i16> %23, i64 0
  %26 = tail call fast float @air.convert.f.f32.u.i16(i16 %25) #6
  %27 = insertelement <4 x float> undef, float %26, i64 0
  %28 = extractelement <2 x i16> %23, i64 1
  %29 = tail call fast float @air.convert.f.f32.u.i16(i16 %28) #6
  %30 = insertelement <4 x float> %27, float %29, i64 1
  %31 = fmul fast float %26, 5.000000e-01
  %32 = insertelement <4 x float> %30, float %31, i64 2
  %33 = fmul fast float %29, 2.500000e-01
  %34 = insertelement <4 x float> %32, float %33, i64 3
  %35 = bitcast ptr addrspace(4) %24 to ptr addrspace(4)
  store <4 x float> %34, ptr addrspace(4) %35, align 16, !tbaa !25
  %36 = fmul fast float %29, 1.600000e+01
  %37 = fadd fast float %36, %26
  %38 = getelementptr inbounds i8, ptr addrspace(4) %24, i64 16
  %39 = bitcast ptr addrspace(4) %38 to ptr addrspace(4)
  store float %37, ptr addrspace(4) %39, align 16, !tbaa !28
  %40 = or <2 x i16> %6, <i16 0, i16 1>
  %41 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %40, i32 0, i16 0) #5
  %42 = extractelement <2 x i16> %40, i64 0
  %43 = tail call fast float @air.convert.f.f32.u.i16(i16 %42) #6
  %44 = insertelement <4 x float> undef, float %43, i64 0
  %45 = extractelement <2 x i16> %40, i64 1
  %46 = tail call fast float @air.convert.f.f32.u.i16(i16 %45) #6
  %47 = insertelement <4 x float> %44, float %46, i64 1
  %48 = fmul fast float %43, 5.000000e-01
  %49 = insertelement <4 x float> %47, float %48, i64 2
  %50 = fmul fast float %46, 2.500000e-01
  %51 = insertelement <4 x float> %49, float %50, i64 3
  %52 = bitcast ptr addrspace(4) %41 to ptr addrspace(4)
  store <4 x float> %51, ptr addrspace(4) %52, align 16, !tbaa !25
  %53 = fmul fast float %46, 1.600000e+01
  %54 = fadd fast float %53, %43
  %55 = getelementptr inbounds i8, ptr addrspace(4) %41, i64 16
  %56 = bitcast ptr addrspace(4) %55 to ptr addrspace(4)
  store float %54, ptr addrspace(4) %56, align 16, !tbaa !28
  %57 = or <2 x i16> %6, splat (i16 1)
  %58 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %57, i32 0, i16 0) #5
  %59 = extractelement <2 x i16> %57, i64 0
  %60 = tail call fast float @air.convert.f.f32.u.i16(i16 %59) #6
  %61 = insertelement <4 x float> undef, float %60, i64 0
  %62 = extractelement <2 x i16> %57, i64 1
  %63 = tail call fast float @air.convert.f.f32.u.i16(i16 %62) #6
  %64 = insertelement <4 x float> %61, float %63, i64 1
  %65 = fmul fast float %60, 5.000000e-01
  %66 = insertelement <4 x float> %64, float %65, i64 2
  %67 = fmul fast float %63, 2.500000e-01
  %68 = insertelement <4 x float> %66, float %67, i64 3
  %69 = bitcast ptr addrspace(4) %58 to ptr addrspace(4)
  store <4 x float> %68, ptr addrspace(4) %69, align 16, !tbaa !25
  %70 = fmul fast float %63, 1.600000e+01
  %71 = fadd fast float %70, %60
  %72 = getelementptr inbounds i8, ptr addrspace(4) %58, i64 16
  %73 = bitcast ptr addrspace(4) %72 to ptr addrspace(4)
  store float %71, ptr addrspace(4) %73, align 16, !tbaa !28
  tail call void @air.wg.barrier(i32 8, i32 1) #7
  %74 = icmp eq <2 x i16> %3, zeroinitializer
  %75 = tail call i1 @air.all.v2i1(<2 x i1> %74) #6
  br i1 %75, label %76, label %79

76:                                               ; preds = %5
  %77 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0) #5
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4f32(ptr addrspace(1) captures(none) %0, ptr addrspace(4) captures(none) %77, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %4, i16 0, i1 false, i32 2) #8
  %78 = getelementptr inbounds i8, ptr addrspace(4) %77, i64 16
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.f32(ptr addrspace(1) captures(none) %1, ptr addrspace(4) captures(none) %78, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %4, i16 0, i1 false, i32 2) #8
  br label %79

79:                                               ; preds = %76, %5
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i16(i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i1 @air.all.v2i1(<2 x i1>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f32(ptr addrspace(1) captures(none), ptr addrspace(4) captures(none), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_imageblock_slice_to_texture_2d.i16.f32(ptr addrspace(1) captures(none), ptr addrspace(4) captures(none), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32) local_unnamed_addr #4

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="32" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent mustprogress nounwind willreturn }
attributes #3 = { mustprogress nofree nounwind willreturn memory(read) }
attributes #4 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nounwind willreturn memory(read) }
attributes #6 = { nounwind willreturn memory(none) }
attributes #7 = { convergent nounwind willreturn }
attributes #8 = { nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!18, !19, !20}
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
!9 = !{ptr @kernel_imageblock_scaled_tile, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !16, !17}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"colourOut"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"weightOut"}
!14 = !{i32 2, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 32, !"air.struct_type_info", !15, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"imageblock<TileCell, layout_explicit>", !"air.arg_name", !"img"}
!15 = !{i32 0, i32 16, i32 0, !"float4", !"colour", i32 16, i32 4, i32 0, !"float", !"weight"}
!16 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}
!17 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
!24 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_imageblock_scaled_tile.metal"}
!25 = !{!26, !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29, !30, i64 16}
!29 = !{!"_ZTS8TileCell", !26, i64 0, !30, i64 16}
!30 = !{!"float", !26, i64 0}
