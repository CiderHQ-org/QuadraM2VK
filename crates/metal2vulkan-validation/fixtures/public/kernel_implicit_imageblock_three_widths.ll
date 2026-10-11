; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. An implicit imageblock
; whose three members are a float, a half and a uint, so one kernel reaches every width of
; `air.load.implicit_imageblock` and `air.store.implicit_imageblock`.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_implicit_imageblock_three_widths.bc'
source_filename = "validation/fixtures/public/kernel_implicit_imageblock_three_widths.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

; Function Attrs: mustprogress nounwind willreturn
define void @kernel_implicit_imageblock_three_widths(%"struct.metal::_imageblock_base" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, <2 x i16> noundef %2) local_unnamed_addr #0 {
  %4 = tail call fast float @air.load.implicit_imageblock.f32(i32 0, <2 x i16> %2, i32 0, i16 0) #3
  %5 = tail call fast half @air.load.implicit_imageblock.f16(i32 1, <2 x i16> %2, i32 0, i16 0) #3
  %6 = tail call i32 @air.load.implicit_imageblock.i32(i32 2, <2 x i16> %2, i32 0, i16 0) #3
  %7 = extractelement <2 x i16> %2, i64 1
  %8 = zext i16 %7 to i32
  %9 = shl nuw nsw i32 %8, 4
  %10 = extractelement <2 x i16> %2, i64 0
  %11 = zext i16 %10 to i32
  %12 = add nuw nsw i32 %9, %11
  %13 = mul nuw nsw i32 %12, 3
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %14
  %16 = bitcast ptr addrspace(1) %15 to ptr addrspace(1)
  store float %4, ptr addrspace(1) %16, align 4, !tbaa !23, !alias.scope !27
  %17 = bitcast half %5 to i16
  %18 = zext i16 %17 to i32
  %19 = add nuw nsw i32 %13, 1
  %20 = zext i32 %19 to i64
  %21 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %20
  store i32 %18, ptr addrspace(1) %21, align 4, !tbaa !23, !alias.scope !27
  %22 = add nuw nsw i32 %13, 2
  %23 = zext i32 %22 to i64
  %24 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %23
  store i32 %6, ptr addrspace(1) %24, align 4, !tbaa !23, !alias.scope !27
  %25 = fadd fast float %4, 1.000000e+00
  %26 = fmul fast half %5, 0xH4000
  %27 = add i32 %6, 100
  tail call void @air.store.implicit_imageblock.f32(float %25, i32 0, <2 x i16> %2, i32 0, i16 0) #4
  tail call void @air.store.implicit_imageblock.f16(half %26, i32 1, <2 x i16> %2, i32 0, i16 0) #4
  tail call void @air.store.implicit_imageblock.i32(i32 %27, i32 2, <2 x i16> %2, i32 0, i16 0) #4
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare float @air.load.implicit_imageblock.f32(i32, <2 x i16>, i32, i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare half @air.load.implicit_imageblock.f16(i32, <2 x i16>, i32, i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare i32 @air.load.implicit_imageblock.i32(i32, <2 x i16>, i32, i16) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn memory(write)
declare void @air.store.implicit_imageblock.f32(float, i32, <2 x i16>, i32, i16) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn memory(write)
declare void @air.store.implicit_imageblock.f16(half, i32, <2 x i16>, i32, i16) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn memory(write)
declare void @air.store.implicit_imageblock.i32(i32, i32, <2 x i16>, i32, i16) local_unnamed_addr #2

attributes #0 = { mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="32" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(read) }
attributes #2 = { mustprogress nounwind willreturn memory(write) }
attributes #3 = { nounwind willreturn memory(read) }
attributes #4 = { nounwind willreturn memory(write) }

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
!9 = !{ptr @kernel_implicit_imageblock_three_widths, !10, !11}
!10 = !{}
!11 = !{!12, !14, !15}
!12 = !{i32 0, !"air.imageblock", !"implicit", !"air.struct_type_info", !13, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"imageblock<Block, layout_implicit>", !"air.arg_name", !"blk"}
!13 = !{i32 0, i32 4, i32 0, !"float", !"scale", !"air.render_target", i32 0, i32 4, i32 2, i32 0, !"half", !"weight", !"air.render_target", i32 1, i32 8, i32 4, i32 0, !"uint", !"tag", !"air.render_target", i32 2}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!15 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"pos"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/private/tmp/ib/k.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(1)"}
!29 = distinct !{!29, !"air-alias-scopes(kernel_implicit_imageblock_three_widths)"}
