; Generated from kernel_threadgroup_flat_leaf.metal by
;   xcrun metal -std=metal3.0 -S -emit-llvm, retripled to spirv-unknown-vulkan1.2, opt -S.
; No hand edits.
;
; The shape this reaches: a `threadgroup Bucket *` KERNEL PARAMETER, so the Workgroup root's
; pointee is an array of records while every load and store through it wants one `uint` leaf. The
; whole-record assignment `tile[tid + 8] = tile[src]` is what leaves a flat one-index access chain
; behind -- the frontend emits it as `llvm.memcpy` between two record pointers carrying no member
; index -- and that is the only shape `rewrite_flattened_workgroup_leaf_accesses` repairs.
; ModuleID = 'k.fix.ll'
source_filename = "k.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

%struct.Bucket = type { i32, i32, i32, i32 }

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_threadgroup_flat_leaf(ptr addrspace(3) captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %2, i32 %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %5
  %7 = load i32, ptr addrspace(1) %6, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %8 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %0, i64 %5, i32 0
  store i32 %7, ptr addrspace(3) %8, align 4, !tbaa !34, !alias.scope !36, !noalias !37
  %9 = add i32 %7, 1
  %10 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %0, i64 %5, i32 1
  store i32 %9, ptr addrspace(3) %10, align 4, !tbaa !38, !alias.scope !36, !noalias !37
  %11 = add i32 %7, 2
  %12 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %0, i64 %5, i32 2
  store i32 %11, ptr addrspace(3) %12, align 4, !tbaa !39, !alias.scope !36, !noalias !37
  %13 = add i32 %7, 3
  %14 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %0, i64 %5, i32 3
  store i32 %13, ptr addrspace(3) %14, align 4, !tbaa !40, !alias.scope !36, !noalias !37
  tail call void @air.wg.barrier(i32 2, i32 1) #3
  %15 = add i32 %3, 1
  %16 = and i32 %15, 7
  %17 = zext i32 %16 to i64
  %18 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %0, i64 %17
  %19 = add i32 %3, 8
  %20 = zext i32 %19 to i64
  %21 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %0, i64 %20
  %22 = bitcast ptr addrspace(3) %21 to ptr addrspace(3)
  %23 = bitcast ptr addrspace(3) %18 to ptr addrspace(3)
  tail call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) noundef align 4 dereferenceable(16) %22, ptr addrspace(3) noundef align 4 dereferenceable(16) %23, i64 16, i1 false), !tbaa.struct !41, !alias.scope !36, !noalias !37
  %24 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %21, i64 0, i32 0
  %25 = load i32, ptr addrspace(3) %24, align 4, !tbaa !34, !alias.scope !36, !noalias !37
  %26 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %0, i64 %20, i32 1
  %27 = load i32, ptr addrspace(3) %26, align 4, !tbaa !38, !alias.scope !36, !noalias !37
  %28 = shl i32 %27, 1
  %29 = add i32 %28, %25
  %30 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %0, i64 %20, i32 2
  %31 = load i32, ptr addrspace(3) %30, align 4, !tbaa !39, !alias.scope !36, !noalias !37
  %32 = shl i32 %31, 2
  %33 = add i32 %29, %32
  %34 = getelementptr inbounds %struct.Bucket, ptr addrspace(3) %0, i64 %20, i32 3
  %35 = load i32, ptr addrspace(3) %34, align 4, !tbaa !40, !alias.scope !36, !noalias !37
  %36 = shl i32 %35, 3
  %37 = add i32 %33, %36
  %38 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %5
  store i32 %37, ptr addrspace(1) %38, align 4, !tbaa !24, !alias.scope !42, !noalias !43
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) noalias writeonly captures(none), ptr addrspace(3) noalias readonly captures(none), i64, i1 immarg) #2

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { convergent nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}
!air.source_file_name = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_threadgroup_flat_leaf, !10, !11}
!10 = !{}
!11 = !{!12, !14, !15, !16}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !13, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Bucket", !"air.arg_name", !"tile"}
!13 = !{i32 0, i32 4, i32 0, !"uint", !"a", i32 4, i32 4, i32 0, !"uint", !"b", i32 8, i32 4, i32 0, !"uint", !"c", i32 12, i32 4, i32 0, !"uint", !"d"}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!15 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!16 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 3, i32 0, i32 0}
!23 = !{!"/private/tmp/fx2/k.metal"}
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(2)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_threadgroup_flat_leaf)"}
!31 = !{!32, !33}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(0)"}
!33 = distinct !{!33, !30, !"air-alias-scope-arg(1)"}
!34 = !{!35, !25, i64 0}
!35 = !{!"_ZTS6Bucket", !25, i64 0, !25, i64 4, !25, i64 8, !25, i64 12}
!36 = !{!32}
!37 = !{!33, !29}
!38 = !{!35, !25, i64 4}
!39 = !{!35, !25, i64 8}
!40 = !{!35, !25, i64 12}
!41 = !{i64 0, i64 4, !24, i64 4, i64 4, !24, i64 8, i64 4, !24, i64 12, i64 4, !24}
!42 = !{!33}
!43 = !{!32, !29}
