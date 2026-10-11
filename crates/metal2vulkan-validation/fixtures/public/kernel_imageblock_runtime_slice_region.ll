; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'ibrsr.bc'
source_filename = "validation/fixtures/public/kernel_imageblock_runtime_slice_region.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_imageblock_runtime_slice_region(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, %"struct.metal::_imageblock_base" %3, <2 x i16> noundef %4, <2 x i16> noundef %5) local_unnamed_addr #0 {
  %7 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %4, i32 0, i16 0) #5
  %8 = extractelement <2 x i16> %4, i64 0
  %9 = tail call fast float @air.convert.f.f32.u.i16(i16 %8) #6
  %10 = extractelement <2 x i16> %4, i64 1
  %11 = tail call fast float @air.convert.f.f32.u.i16(i16 %10) #6
  %12 = insertelement <4 x float> undef, float %9, i64 0
  %13 = insertelement <4 x float> %12, float %11, i64 1
  %14 = fmul fast float %9, 5.000000e-01
  %15 = fadd fast float %14, 2.500000e-01
  %16 = insertelement <4 x float> %13, float %15, i64 2
  %17 = fmul fast float %11, 2.500000e-01
  %18 = fadd fast float %17, 5.000000e-01
  %19 = insertelement <4 x float> %16, float %18, i64 3
  %20 = bitcast ptr addrspace(4) %7 to ptr addrspace(4)
  store <4 x float> %19, ptr addrspace(4) %20, align 16, !tbaa !26
  %21 = mul i16 %8, 7
  %22 = add i16 %21, -3
  %23 = insertelement <4 x i16> undef, i16 %22, i64 0
  %24 = mul i16 %10, 11
  %25 = add i16 %24, -5
  %26 = insertelement <4 x i16> %23, i16 %25, i64 1
  %27 = mul i16 %8, 100
  %28 = add i16 %27, %10
  %29 = insertelement <4 x i16> %26, i16 %28, i64 2
  %30 = mul i16 %10, -16
  %31 = sub i16 %30, %8
  %32 = insertelement <4 x i16> %29, i16 %31, i64 3
  %33 = getelementptr inbounds i8, ptr addrspace(4) %7, i64 16
  %34 = bitcast ptr addrspace(4) %33 to ptr addrspace(4)
  store <4 x i16> %32, ptr addrspace(4) %34, align 16, !tbaa !26
  %35 = tail call fast half @air.convert.f.f16.u.i16(i16 %8) #6
  %36 = tail call fast half @air.convert.f.f16.u.i16(i16 %10) #6
  %37 = fmul fast half %36, 0xH4800
  %38 = fadd fast half %37, %35
  %39 = getelementptr inbounds i8, ptr addrspace(4) %7, i64 24
  %40 = bitcast ptr addrspace(4) %39 to ptr addrspace(4)
  store half %38, ptr addrspace(4) %40, align 8, !tbaa !29
  tail call void @air.wg.barrier(i32 8, i32 1) #7
  %41 = icmp eq <2 x i16> %4, zeroinitializer
  %42 = tail call i1 @air.all.v2i1(<2 x i1> %41) #6
  br i1 %42, label %43, label %54

43:                                               ; preds = %6
  %44 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0) #5
  %45 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> splat (i16 4), i32 0, i16 0) #5
  tail call void @air.write_imageblock_slice_to_texture_2d.v4f32(ptr addrspace(1) captures(none) %0, ptr addrspace(4) captures(none) %44, i1 true, <2 x i16> zeroinitializer, <2 x i16> %5, <2 x i32> zeroinitializer, i32 0, i1 false, i32 2) #8
  %46 = getelementptr inbounds i8, ptr addrspace(4) %44, i64 16
  %47 = lshr <2 x i16> %5, splat (i16 1)
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4i16(ptr addrspace(1) captures(none) %1, ptr addrspace(4) captures(none) %46, i1 true, <2 x i16> zeroinitializer, <2 x i16> %47, <2 x i16> <i16 8, i16 0>, i16 0, i1 false, i32 2) #8
  %48 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> <i16 5, i16 3>, i32 0, i16 0) #5
  %49 = getelementptr inbounds i8, ptr addrspace(4) %45, i64 24
  tail call void @air.write_imageblock_slice_to_texture_2d.f16(ptr addrspace(1) captures(none) %2, ptr addrspace(4) captures(none) %49, i1 true, <2 x i16> zeroinitializer, <2 x i16> %47, <2 x i32> <i32 0, i32 8>, i32 0, i1 false, i32 2) #8
  %50 = getelementptr inbounds i8, ptr addrspace(4) %44, i64 24
  %51 = lshr <2 x i16> %5, splat (i16 2)
  tail call void @air.write_imageblock_slice_to_texture_2d.f16(ptr addrspace(1) captures(none) %2, ptr addrspace(4) captures(none) %50, i1 true, <2 x i16> zeroinitializer, <2 x i16> %51, <2 x i32> splat (i32 8), i32 0, i1 false, i32 2) #8
  %52 = getelementptr inbounds i8, ptr addrspace(4) %48, i64 24
  %53 = lshr <2 x i16> %5, splat (i16 3)
  tail call void @air.write_imageblock_slice_to_texture_2d.f16(ptr addrspace(1) captures(none) %2, ptr addrspace(4) captures(none) %52, i1 true, <2 x i16> zeroinitializer, <2 x i16> %53, <2 x i32> splat (i32 12), i32 0, i1 false, i32 2) #8
  br label %54

54:                                               ; preds = %43, %6
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i16(i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i16(i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i1 @air.all.v2i1(<2 x i1>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_imageblock_slice_to_texture_2d.v4f32(ptr addrspace(1) captures(none), ptr addrspace(4) captures(none), i1, <2 x i16>, <2 x i16>, <2 x i32>, i32, i1, i32) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4i16(ptr addrspace(1) captures(none), ptr addrspace(4) captures(none), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_imageblock_slice_to_texture_2d.f16(ptr addrspace(1) captures(none), ptr addrspace(4) captures(none), i1, <2 x i16>, <2 x i16>, <2 x i32>, i32, i1, i32) local_unnamed_addr #4

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_imageblock_runtime_slice_region, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !17, !18}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"colourOut"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<short, write>", !"air.arg_name", !"codeOut"}
!14 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"weightOut"}
!15 = !{i32 3, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 32, !"air.struct_type_info", !16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"imageblock<TileCell, layout_explicit>", !"air.arg_name", !"img"}
!16 = !{i32 0, i32 16, i32 0, !"float4", !"colour", i32 16, i32 8, i32 0, !"short4", !"code", i32 24, i32 2, i32 0, !"half", !"weight"}
!17 = !{i32 4, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}
!18 = !{i32 5, !"air.threads_per_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tgSize"}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 4, i32 0, i32 0}
!25 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_imageblock_runtime_slice_region.metal"}
!26 = !{!27, !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C++ TBAA"}
!29 = !{!30, !31, i64 24}
!30 = !{!"_ZTS8TileCell", !27, i64 0, !27, i64 16, !31, i64 24}
!31 = !{!"half", !27, i64 0}
