; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'pkedge.bc'
source_filename = "validation/fixtures/public/kernel_pack_normalized_edges.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_pack_normalized_edges(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = shl i32 %2, 2
  %5 = zext i32 %4 to i64
  %6 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %5
  %7 = load float, ptr addrspace(1) %6, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %8 = insertelement <4 x float> undef, float %7, i64 0
  %9 = or i32 %4, 1
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %10
  %12 = load float, ptr addrspace(1) %11, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %13 = insertelement <4 x float> %8, float %12, i64 1
  %14 = or i32 %4, 2
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %15
  %17 = load float, ptr addrspace(1) %16, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %18 = insertelement <4 x float> %13, float %17, i64 2
  %19 = or i32 %4, 3
  %20 = zext i32 %19 to i64
  %21 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %20
  %22 = load float, ptr addrspace(1) %21, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %23 = insertelement <4 x float> %18, float %22, i64 3
  %24 = mul i32 %2, 7
  %25 = tail call i32 @air.pack.unorm4x8.v4f32(<4 x float> %23) #2
  %26 = zext i32 %24 to i64
  %27 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %26
  store i32 %25, ptr addrspace(1) %27, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %28 = tail call i32 @air.pack.snorm4x8.v4f32(<4 x float> %23) #2
  %29 = add i32 %24, 1
  %30 = zext i32 %29 to i64
  %31 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %30
  store i32 %28, ptr addrspace(1) %31, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %32 = shufflevector <4 x float> %13, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %33 = tail call i32 @air.pack.unorm2x16.v2f32(<2 x float> %32) #2
  %34 = add i32 %24, 2
  %35 = zext i32 %34 to i64
  %36 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %35
  store i32 %33, ptr addrspace(1) %36, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %37 = tail call i32 @air.pack.snorm2x16.v2f32(<2 x float> %32) #2
  %38 = add i32 %24, 3
  %39 = zext i32 %38 to i64
  %40 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %39
  store i32 %37, ptr addrspace(1) %40, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %41 = tail call i32 @air.pack.unorm.rgb10a2.v4f32(<4 x float> %23) #2
  %42 = add i32 %24, 4
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %43
  store i32 %41, ptr addrspace(1) %44, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %45 = shufflevector <4 x float> %18, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %46 = tail call i16 @air.pack.unorm.rgb565.v3f32(<3 x float> %45) #2
  %47 = zext i16 %46 to i32
  %48 = add i32 %24, 5
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %49
  store i32 %47, ptr addrspace(1) %50, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %51 = tail call i32 @air.pack.unorm4x8.srgb.v4f32(<4 x float> %23) #2
  %52 = add i32 %24, 6
  %53 = zext i32 %52 to i64
  %54 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %53
  store i32 %51, ptr addrspace(1) %54, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.pack.unorm4x8.v4f32(<4 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.pack.snorm4x8.v4f32(<4 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.pack.unorm2x16.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.pack.snorm2x16.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.pack.unorm.rgb10a2.v4f32(<4 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.pack.unorm.rgb565.v3f32(<3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.pack.unorm4x8.srgb.v4f32(<4 x float>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_pack_normalized_edges, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_pack_normalized_edges.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_pack_normalized_edges)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !24, i64 0}
