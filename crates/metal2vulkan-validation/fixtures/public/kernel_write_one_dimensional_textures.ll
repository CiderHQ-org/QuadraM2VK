; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'f.bc'
source_filename = "validation/fixtures/public/kernel_write_one_dimensional_textures.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nounwind willreturn
define void @kernel_write_one_dimensional_textures(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = mul i32 %2, 3
  %5 = add i32 %4, 16
  %6 = insertelement <4 x i32> undef, i32 %5, i64 0
  %7 = mul i32 %2, 5
  %8 = sub i32 240, %7
  %9 = insertelement <4 x i32> %6, i32 %8, i64 1
  %10 = mul i32 %2, 7
  %11 = add i32 %10, 65
  %12 = insertelement <4 x i32> %9, i32 %11, i64 2
  %13 = xor i32 %2, 144
  %14 = insertelement <4 x i32> %12, i32 %13, i64 3
  tail call void @air.write_texture_1d.u.v4i32(ptr addrspace(1) captures(none) %0, i32 %2, <4 x i32> %14, i32 0, i32 2) #2, !alias.scope !22
  %15 = add i32 %7, 32
  %16 = insertelement <4 x i32> undef, i32 %15, i64 0
  %17 = sub i32 225, %4
  %18 = insertelement <4 x i32> %16, i32 %17, i64 1
  %19 = mul i32 %2, 11
  %20 = add i32 %19, 82
  %21 = insertelement <4 x i32> %18, i32 %20, i64 2
  %22 = shl i32 %2, 1
  %23 = xor i32 %22, 112
  %24 = insertelement <4 x i32> %21, i32 %23, i64 3
  tail call void @air.write_texture_buffer_1d.u.v4i32(ptr addrspace(1) captures(none) %1, i32 %2, <4 x i32> %24, i32 2) #2, !alias.scope !22
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_1d.u.v4i32(ptr addrspace(1) captures(none), i32, <4 x i32>, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_buffer_1d.u.v4i32(ptr addrspace(1) captures(none), i32, <4 x i32>, i32) local_unnamed_addr #1

attributes #0 = { mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_write_one_dimensional_textures, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture1d<uint, write>", !"air.arg_name", !"line"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture_buffer<uint, write>", !"air.arg_name", !"texels"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_write_one_dimensional_textures.metal"}
!22 = !{!23}
!23 = distinct !{!23, !24, !"air-alias-scope-textures"}
!24 = distinct !{!24, !"air-alias-scopes(kernel_write_one_dimensional_textures)"}
