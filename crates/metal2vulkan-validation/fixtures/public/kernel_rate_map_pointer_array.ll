; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -x metal
; -S -emit-llvm -` (stdin) and round-tripped through LLVM 21's llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_rate_map_pointer_array.bc'
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@_ZL7kScreen = internal unnamed_addr addrspace(2) constant [8 x <2 x float>] [<2 x float> zeroinitializer, <2 x float> <float 1.500000e+00, float 2.500000e+00>, <2 x float> <float 1.025000e+01, float 7.750000e+00>, <2 x float> <float 1.750000e+01, float 1.150000e+01>, <2 x float> <float 2.012500e+01, float 3.875000e+00>, <2 x float> <float 3.125000e+01, float 2.075000e+01>, <2 x float> <float 3.550000e+01, float 2.150000e+01>, <2 x float> <float 5.000000e+00, float 1.800000e+01>], align 8

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: write)
define void @kernel_rate_map_pointer_array(ptr addrspace(2) noundef align 1 "air-buffer-no-alias" %0, ptr addrspace(2) noundef align 1 "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, <2 x i32> noundef %3) local_unnamed_addr #0 {
  %5 = alloca [2 x ptr addrspace(2)], align 8
  %6 = bitcast ptr %5 to ptr
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %6) #3
  %7 = getelementptr inbounds [2 x ptr addrspace(2)], ptr %5, i64 0, i64 0
  store ptr addrspace(2) %0, ptr %7, align 8, !tbaa !22
  %8 = getelementptr inbounds [2 x ptr addrspace(2)], ptr %5, i64 0, i64 1
  store ptr addrspace(2) %1, ptr %8, align 8, !tbaa !22
  %9 = extractelement <2 x i32> %3, i64 1
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds [2 x ptr addrspace(2)], ptr %5, i64 0, i64 %10
  %12 = bitcast ptr %11 to ptr
  %13 = load ptr addrspace(2), ptr %12, align 8, !tbaa !22
  %14 = extractelement <2 x i32> %3, i64 0
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds [8 x <2 x float>], ptr addrspace(2) @_ZL7kScreen, i64 0, i64 %15
  %17 = load <2 x float>, ptr addrspace(2) %16, align 8, !tbaa !26
  %18 = tail call fast <2 x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> %17, ptr addrspace(2) readonly captures(none) %13, i32 0) #4
  %19 = shl i32 %9, 3
  %20 = add i32 %19, %14
  %21 = bitcast <2 x float> %18 to <2 x i32>
  %22 = extractelement <2 x i32> %21, i64 0
  %23 = shl i32 %20, 1
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %24
  store i32 %22, ptr addrspace(1) %25, align 4, !tbaa !27, !alias.scope !29, !noalias !32
  %26 = extractelement <2 x i32> %21, i64 1
  %27 = or i32 %23, 1
  %28 = zext i32 %27 to i64
  %29 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %28
  store i32 %26, ptr addrspace(1) %29, align 4, !tbaa !27, !alias.scope !29, !noalias !32
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %6) #3
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare <2 x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2) readonly captures(none), i32) local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #2

attributes #0 = { mustprogress nofree nounwind willreturn memory(argmem: write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nounwind }
attributes #4 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
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
!9 = !{ptr @kernel_rate_map_pointer_array, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"metal::rasterization_rate_map_data", !"air.arg_name", !"left"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"metal::rasterization_rate_map_data", !"air.arg_name", !"right"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tid"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!23, !23, i64 0}
!23 = !{!"any pointer", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!24, !24, i64 0}
!27 = !{!28, !28, i64 0}
!28 = !{!"int", !24, i64 0}
!29 = !{!30}
!30 = distinct !{!30, !31, !"air-alias-scope-arg(2)"}
!31 = distinct !{!31, !"air-alias-scopes(kernel_rate_map_pointer_array)"}
!32 = !{!33, !34}
!33 = distinct !{!33, !31, !"air-alias-scope-arg(0)"}
!34 = distinct !{!34, !31, !"air-alias-scope-arg(1)"}
