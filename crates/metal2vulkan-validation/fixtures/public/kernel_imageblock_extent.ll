; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'ie.bc'
source_filename = "validation/fixtures/public/kernel_imageblock_extent.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

; Function Attrs: mustprogress nofree nounwind willreturn
define void @kernel_imageblock_extent(%"struct.metal::_imageblock_base" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, <2 x i16> noundef %2) local_unnamed_addr #0 {
  %4 = extractelement <2 x i16> %2, i64 1
  %5 = zext i16 %4 to i32
  %6 = shl nuw nsw i32 %5, 4
  %7 = extractelement <2 x i16> %2, i64 0
  %8 = zext i16 %7 to i32
  %9 = add nuw nsw i32 %6, %8
  %10 = shl nuw nsw i32 %9, 2
  %11 = tail call i16 @air.get_imageblock_width() #3
  %12 = zext i16 %11 to i32
  %13 = zext i32 %10 to i64
  %14 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %13
  store i32 %12, ptr addrspace(1) %14, align 4, !tbaa !23, !alias.scope !27
  %15 = tail call i16 @air.get_imageblock_height() #3
  %16 = zext i16 %15 to i32
  %17 = or i32 %10, 1
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %18
  store i32 %16, ptr addrspace(1) %19, align 4, !tbaa !23, !alias.scope !27
  %20 = tail call fast <4 x half> @air.load.implicit_imageblock.v4f16(i32 0, <2 x i16> %2, i32 0, i16 0) #4
  %21 = bitcast <4 x half> %20 to <4 x i16>
  %22 = extractelement <4 x i16> %21, i64 0
  %23 = zext i16 %22 to i32
  %24 = or i32 %10, 2
  %25 = zext i32 %24 to i64
  %26 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %25
  store i32 %23, ptr addrspace(1) %26, align 4, !tbaa !23, !alias.scope !27
  %27 = tail call fast <4 x half> @air.load.implicit_imageblock.v4f16(i32 0, <2 x i16> %2, i32 0, i16 0) #4
  %28 = bitcast <4 x half> %27 to <4 x i16>
  %29 = extractelement <4 x i16> %28, i64 2
  %30 = zext i16 %29 to i32
  %31 = or i32 %10, 3
  %32 = zext i32 %31 to i64
  %33 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %32
  store i32 %30, ptr addrspace(1) %33, align 4, !tbaa !23, !alias.scope !27
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.get_imageblock_width() local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.get_imageblock_height() local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare <4 x half> @air.load.implicit_imageblock.v4f16(i32, <2 x i16>, i32, i16) local_unnamed_addr #2

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(read) }
attributes #3 = { nounwind willreturn memory(none) }
attributes #4 = { nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!16, !17, !18}
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
!9 = !{ptr @kernel_imageblock_extent, !10, !11}
!10 = !{}
!11 = !{!12, !14, !15}
!12 = !{i32 0, !"air.imageblock", !"implicit", !"air.struct_type_info", !13, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<IBColor, layout_implicit>", !"air.arg_name", !"block"}
!13 = !{i32 0, i32 8, i32 0, !"half4", !"value", !"air.render_target", i32 0}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!15 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"pos"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_imageblock_extent.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(1)"}
!29 = distinct !{!29, !"air-alias-scopes(kernel_imageblock_extent)"}
