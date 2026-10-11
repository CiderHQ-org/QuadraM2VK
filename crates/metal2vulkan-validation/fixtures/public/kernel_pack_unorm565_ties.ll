; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'p565.bc'
source_filename = "validation/fixtures/public/kernel_pack_unorm565_ties.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_pack_unorm565_ties(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds <4 x float>, ptr addrspace(1) %0, i64 %5
  %7 = load <4 x float>, ptr addrspace(1) %6, align 16, !tbaa !23, !alias.scope !26, !noalias !29
  %8 = getelementptr inbounds <4 x i16>, ptr addrspace(1) %1, i64 %5
  %9 = bitcast ptr addrspace(1) %8 to ptr addrspace(1)
  %10 = load <4 x half>, ptr addrspace(1) %9, align 8, !tbaa !23, !alias.scope !32, !noalias !33
  %11 = shl i32 %3, 1
  %12 = shufflevector <4 x float> %7, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %13 = tail call i16 @air.pack.unorm.rgb565.v3f32(<3 x float> %12) #2
  %14 = zext i16 %13 to i32
  %15 = zext i32 %11 to i64
  %16 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %15
  store i32 %14, ptr addrspace(1) %16, align 4, !tbaa !34, !alias.scope !36, !noalias !37
  %17 = shufflevector <4 x half> %10, <4 x half> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %18 = tail call i16 @air.pack.unorm.rgb565.v3f16(<3 x half> %17) #2
  %19 = zext i16 %18 to i32
  %20 = or i32 %11, 1
  %21 = zext i32 %20 to i64
  %22 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %21
  store i32 %19, ptr addrspace(1) %22, align 4, !tbaa !34, !alias.scope !36, !noalias !37
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.pack.unorm.rgb565.v3f32(<3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.pack.unorm.rgb565.v3f16(<3 x half>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_pack_unorm565_ties, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"fin"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"hin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_pack_unorm565_ties.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_pack_unorm565_ties)"}
!29 = !{!30, !31}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = distinct !{!31, !28, !"air-alias-scope-arg(2)"}
!32 = !{!30}
!33 = !{!27, !31}
!34 = !{!35, !35, i64 0}
!35 = !{!"int", !24, i64 0}
!36 = !{!31}
!37 = !{!27, !30}
