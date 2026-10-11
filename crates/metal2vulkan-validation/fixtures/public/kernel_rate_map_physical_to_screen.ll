; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'rmpts.bc'
source_filename = "validation/fixtures/public/kernel_rate_map_physical_to_screen.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@_ZL9kPhysical = internal unnamed_addr addrspace(2) constant [8 x <2 x float>] [<2 x float> zeroinitializer, <2 x float> <float 1.500000e+00, float 2.500000e+00>, <2 x float> <float 3.125000e+01, float 4.775000e+01>, <2 x float> splat (float 6.350000e+01), <2 x float> splat (float 6.400000e+01), <2 x float> <float 8.000000e+01, float 7.250000e+00>, <2 x float> <float -3.500000e+00, float 2.000000e+00>, <2 x float> <float 8.125000e+00, float 5.587500e+01>], align 8

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: write)
define void @kernel_rate_map_physical_to_screen(ptr addrspace(2) noundef readonly align 1 captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds [8 x <2 x float>], ptr addrspace(2) @_ZL9kPhysical, i64 0, i64 %4
  %6 = load <2 x float>, ptr addrspace(2) %5, align 8, !tbaa !22
  %7 = bitcast ptr addrspace(2) %0 to ptr addrspace(2)
  %8 = tail call fast <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float> %6, ptr addrspace(2) readonly captures(none) %7, i32 0) #2, !alias.scope !25, !noalias !28
  %9 = bitcast <2 x float> %8 to <2 x i32>
  %10 = extractelement <2 x i32> %9, i64 0
  %11 = shl i32 %2, 1
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %12
  store i32 %10, ptr addrspace(1) %13, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %14 = extractelement <2 x i32> %9, i64 1
  %15 = or i32 %11, 1
  %16 = zext i32 %15 to i64
  %17 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %16
  store i32 %14, ptr addrspace(1) %17, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2) readonly captures(none), i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind willreturn memory(argmem: write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind willreturn memory(argmem: read) }

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
!9 = !{ptr @kernel_rate_map_physical_to_screen, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"metal::rasterization_rate_map_data", !"air.arg_name", !"map"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_rate_map_physical_to_screen.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_rate_map_physical_to_screen)"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"air-alias-scope-arg(1)"}
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !23, i64 0}
