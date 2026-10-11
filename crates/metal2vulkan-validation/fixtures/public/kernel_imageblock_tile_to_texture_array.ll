; ModuleID = '/tmp/ibarr/f.bc'
source_filename = "validation/fixtures/public/kernel_imageblock_tile_to_texture_array.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_imageblock_tile_to_texture_array(ptr addrspace(1) %0, ptr addrspace(1) %1, %"struct.metal::_imageblock_base" %2, <2 x i16> noundef %3, <2 x i16> noundef %4) local_unnamed_addr #0 {
  %6 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %3, i32 0, i16 0) #5
  %7 = extractelement <2 x i16> %3, i64 0
  %8 = tail call fast half @air.convert.f.f16.u.i16(i16 %7) #6
  %9 = insertelement <4 x half> undef, half %8, i64 0
  %10 = extractelement <2 x i16> %3, i64 1
  %11 = tail call fast half @air.convert.f.f16.u.i16(i16 %10) #6
  %12 = insertelement <4 x half> %9, half %11, i64 1
  %13 = fmul fast half %8, 0xH3800
  %14 = insertelement <4 x half> %12, half %13, i64 2
  %15 = fmul fast half %11, 0xH3400
  %16 = insertelement <4 x half> %14, half %15, i64 3
  %17 = bitcast ptr addrspace(4) %6 to ptr addrspace(4)
  store <4 x half> %16, ptr addrspace(4) %17, align 8, !tbaa !25
  %18 = fmul fast half %11, 0xH5000
  %19 = fadd fast half %18, %8
  %20 = getelementptr inbounds i8, ptr addrspace(4) %6, i64 8
  %21 = bitcast ptr addrspace(4) %20 to ptr addrspace(4)
  store half %19, ptr addrspace(4) %21, align 8, !tbaa !28
  tail call void @air.wg.barrier(i32 8, i32 1) #7
  %22 = icmp eq <2 x i16> %3, zeroinitializer
  %23 = tail call i1 @air.all.v2i1(<2 x i1> %22) #6
  br i1 %23, label %24, label %25

24:                                               ; preds = %5
  tail call void @air.write_imageblock_slice_to_texture_2d_array.i16.v4f16(ptr addrspace(1) captures(none) %0, ptr addrspace(4) captures(none) %6, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %4, i16 2, i16 0, i1 false, i32 2) #8
  tail call void @air.write_imageblock_slice_to_texture_2d_array.i16.f16(ptr addrspace(1) captures(none) %1, ptr addrspace(4) captures(none) %20, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %4, i16 1, i16 0, i1 false, i32 2) #8
  br label %25

25:                                               ; preds = %24, %5
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i16(i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i1 @air.all.v2i1(<2 x i1>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_imageblock_slice_to_texture_2d_array.i16.v4f16(ptr addrspace(1) captures(none), ptr addrspace(4) captures(none), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i16, i1, i32) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_imageblock_slice_to_texture_2d_array.i16.f16(ptr addrspace(1) captures(none), ptr addrspace(4) captures(none), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i16, i1, i32) local_unnamed_addr #4

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
!9 = !{ptr @kernel_imageblock_tile_to_texture_array, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !16, !17}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<half, write>", !"air.arg_name", !"colourOut"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<half, write>", !"air.arg_name", !"weightOut"}
!14 = !{i32 2, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !15, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ArrCell, layout_explicit>", !"air.arg_name", !"img"}
!15 = !{i32 0, i32 8, i32 0, !"half4", !"colour", i32 8, i32 2, i32 0, !"half", !"weight"}
!16 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}
!17 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 3, i32 2, i32 0}
!24 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_imageblock_tile_to_texture_array.metal"}
!25 = !{!26, !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29, !30, i64 8}
!29 = !{!"_ZTS7ArrCell", !26, i64 0, !30, i64 8}
!30 = !{!"half", !26, i64 0}
