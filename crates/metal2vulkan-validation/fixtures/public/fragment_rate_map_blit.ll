; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -x metal
; -S -emit-llvm -` (stdin) and round-tripped through LLVM 21's llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'fragment_rate_map_blit.bc'
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
define i32 @fragment_rate_map_blit(<4 x float> noundef %0, ptr addrspace(2) noundef readonly align 1 captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) %2) local_unnamed_addr #0 {
  %4 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %5 = bitcast ptr addrspace(2) %1 to ptr addrspace(2)
  %6 = tail call fast <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float> %4, ptr addrspace(2) readonly captures(none) %5, i32 0) #4, !alias.scope !22, !noalias !25
  %7 = tail call fast <2 x float> @air.fast_floor.v2f32(<2 x float> %6) #5
  %8 = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none) %2, i32 0) #4, !alias.scope !25, !noalias !22
  %9 = add i32 %8, -1
  %10 = tail call fast float @air.convert.f.f32.u.i32(i32 %9) #5
  %11 = insertelement <2 x float> undef, float %10, i64 0
  %12 = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) readonly captures(none) %2, i32 0) #4, !alias.scope !25, !noalias !22
  %13 = add i32 %12, -1
  %14 = tail call fast float @air.convert.f.f32.u.i32(i32 %13) #5
  %15 = insertelement <2 x float> %11, float %14, i64 1
  %16 = tail call fast <2 x float> @air.fast_clamp.v2f32(<2 x float> %7, <2 x float> zeroinitializer, <2 x float> %15) #5
  %17 = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %16) #5
  %18 = tail call ptr addrspace(2) @air.get_read_sampler() #6
  %19 = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none) %2, ptr addrspace(2) %18, <2 x i32> %17, <2 x i32> zeroinitializer, i32 0, i32 1) #4
  %20 = extractvalue { <4 x i32>, i8 } %19, 0
  %21 = extractelement <4 x i32> %20, i64 0
  ret i32 %21
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_floor.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2) readonly captures(none), i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_clamp.v2f32(<2 x float>, <2 x float>, <2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_height_texture_2d(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: read)
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #2

attributes #0 = { mustprogress nofree nounwind willreturn memory(read) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #4 = { nounwind willreturn memory(argmem: read) }
attributes #5 = { nounwind willreturn memory(none) }
attributes #6 = { nounwind willreturn memory(inaccessiblemem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.fragment = !{!9}
!air.compile_options = !{!16, !17, !18}
!llvm.ident = !{!19}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @fragment_rate_map_blit, !10, !12}
!10 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint"}
!12 = !{!13, !14, !15}
!13 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"metal::rasterization_rate_map_data", !"air.arg_name", !"map"}
!15 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"screen"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!23}
!23 = distinct !{!23, !24, !"air-alias-scope-arg(1)"}
!24 = distinct !{!24, !"air-alias-scopes(fragment_rate_map_blit)"}
!25 = !{!26}
!26 = distinct !{!26, !24, !"air-alias-scope-textures"}
