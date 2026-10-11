; ModuleID = '/tmp/mkfix2/fx.bc'
source_filename = "validation/fixtures/public/kernel_write_short_coordinate_textures.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nounwind
define void @kernel_write_short_coordinate_textures(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, <2 x i16> noundef %3) local_unnamed_addr #0 {
  %5 = extractelement <2 x i16> %3, i64 1
  %6 = shl i16 %5, 2
  %7 = extractelement <2 x i16> %3, i64 0
  %8 = zext i16 %7 to i32
  %9 = shl nuw i32 %8, 16
  %10 = add i32 %9, 65536
  %11 = add i16 %7, 256
  %12 = add i16 %5, 300
  %13 = add i16 %6, %7
  br label %25

14:                                               ; preds = %25
  %15 = shl i16 %5, 2
  %16 = add i16 %15, %7
  %17 = xor i16 %16, -32768
  %18 = insertelement <4 x i16> undef, i16 %17, i64 0
  %19 = sub i16 -129, %7
  %20 = insertelement <4 x i16> %18, i16 %19, i64 1
  %21 = add i16 %5, 128
  %22 = insertelement <4 x i16> %20, i16 %21, i64 2
  %23 = sub i16 32767, %16
  %24 = insertelement <4 x i16> %22, i16 %23, i64 3
  tail call void @air.write_texture_2d.i16.s.v4i16(ptr addrspace(1) captures(none) %2, <2 x i16> %3, <4 x i16> %24, i16 0, i32 2) #2, !alias.scope !23
  ret void

25:                                               ; preds = %25, %4
  %26 = phi i32 [ 0, %4 ], [ %45, %25 ]
  %27 = trunc i32 %26 to i16
  %28 = shl i16 %27, 4
  %29 = add i16 %13, %28
  %30 = zext i16 %29 to i32
  %31 = xor i32 %30, -559038737
  %32 = insertelement <4 x i32> undef, i32 %31, i64 0
  %33 = or i32 %30, -2147483648
  %34 = insertelement <4 x i32> %32, i32 %33, i64 1
  %35 = insertelement <4 x i32> %34, i32 %10, i64 2
  %36 = xor i32 %30, 2147483647
  %37 = insertelement <4 x i32> %35, i32 %36, i64 3
  %38 = trunc i32 %26 to i16
  tail call void @air.write_texture_2d_array.i16.u.v4i32(ptr addrspace(1) captures(none) %0, <2 x i16> %3, i16 %38, <4 x i32> %37, i16 0, i32 2) #2, !alias.scope !23
  %39 = xor i16 %29, -1
  %40 = insertelement <4 x i16> undef, i16 %39, i64 0
  %41 = xor i16 %29, -32768
  %42 = insertelement <4 x i16> %40, i16 %41, i64 1
  %43 = insertelement <4 x i16> %42, i16 %11, i64 2
  %44 = insertelement <4 x i16> %43, i16 %12, i64 3
  tail call void @air.write_texture_2d_array.i16.u.v4i16(ptr addrspace(1) captures(none) %1, <2 x i16> %3, i16 %38, <4 x i16> %44, i16 0, i32 2) #2, !alias.scope !23
  %45 = add nuw nsw i32 %26, 1
  %46 = icmp eq i32 %45, 3
  br i1 %46, label %14, label %25, !llvm.loop !26
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_2d_array.i16.u.v4i32(ptr addrspace(1) captures(none), <2 x i16>, i16, <4 x i32>, i16, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_2d_array.i16.u.v4i16(ptr addrspace(1) captures(none), <2 x i16>, i16, <4 x i16>, i16, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_2d.i16.s.v4i16(ptr addrspace(1) captures(none), <2 x i16>, <4 x i16>, i16, i32) local_unnamed_addr #1

attributes #0 = { mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn memory(argmem: readwrite) }

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
!9 = !{ptr @kernel_write_short_coordinate_textures, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<uint, write>", !"air.arg_name", !"wide"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<ushort, write>", !"air.arg_name", !"narrow"}
!14 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<short, write>", !"air.arg_name", !"signed_plane"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_write_short_coordinate_textures.metal"}
!23 = !{!24}
!24 = distinct !{!24, !25, !"air-alias-scope-textures"}
!25 = distinct !{!25, !"air-alias-scopes(kernel_write_short_coordinate_textures)"}
!26 = distinct !{!26, !27}
!27 = !{!"llvm.loop.mustprogress"}
