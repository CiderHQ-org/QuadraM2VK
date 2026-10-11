; ModuleID = 'kernel_divide_shapes.ll'
source_filename = "kernel_divide_shapes.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define void @divshapes(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %2, i32 %3) local_unnamed_addr #0 {
  %5 = zext i32 %3 to i64
  %6 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %5
  %7 = load i32, ptr addrspace(1) %6, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %8 = add i32 %3, 8
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %9
  %11 = load i32, ptr addrspace(1) %10, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %12 = udiv i32 %7, %11
  %13 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %5
  store i32 %12, ptr addrspace(1) %13, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %14 = urem i32 %7, %11
  %15 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %9
  store i32 %14, ptr addrspace(1) %15, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %16 = sdiv i32 %7, %11
  %17 = add i32 %3, 16
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %18
  store i32 %16, ptr addrspace(1) %19, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %20 = srem i32 %7, %11
  %21 = add i32 %3, 24
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %22
  store i32 %20, ptr addrspace(1) %23, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %24 = insertelement <2 x i32> undef, i32 %7, i64 0
  %25 = lshr i32 %7, 3
  %26 = insertelement <2 x i32> %24, i32 %25, i64 1
  %27 = insertelement <2 x i32> undef, i32 %11, i64 0
  %28 = or i32 %11, 1
  %29 = insertelement <2 x i32> %27, i32 %28, i64 1
  %30 = udiv <2 x i32> %26, %29
  %31 = extractelement <2 x i32> %30, i64 0
  %32 = add i32 %3, 32
  %33 = zext i32 %32 to i64
  %34 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %33
  store i32 %31, ptr addrspace(1) %34, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %35 = extractelement <2 x i32> %30, i64 1
  %36 = add i32 %3, 40
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %37
  store i32 %35, ptr addrspace(1) %38, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %39 = zext i32 %7 to i64
  %40 = shl nuw nsw i64 %39, 20
  %41 = zext i32 %11 to i64
  %42 = or i64 %40, %41
  %43 = add nuw nsw i64 %41, 1
  %44 = udiv i64 %42, %43
  %45 = trunc i64 %44 to i32
  %46 = add i32 %3, 48
  %47 = zext i32 %46 to i64
  %48 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %47
  store i32 %45, ptr addrspace(1) %48, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %49 = urem i64 %42, %43
  %50 = trunc i64 %49 to i32
  %51 = add i32 %3, 56
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %52
  store i32 %50, ptr addrspace(1) %53, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %54 = shl i32 %7, 2
  %55 = and i32 %54, 28
  %56 = zext i32 %55 to i64
  %57 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 %56
  %58 = bitcast ptr addrspace(1) %57 to ptr addrspace(1)
  %59 = load i32, ptr addrspace(1) %58, align 4, !tbaa !23, !alias.scope !35, !noalias !36
  %60 = add i32 %3, 64
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %61
  store i32 %59, ptr addrspace(1) %62, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

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
!9 = !{ptr @divshapes, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"raw"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 3, i32 0, i32 0}
!22 = !{!"/private/tmp/divcase/kernel_divide_shapes.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(0)"}
!29 = distinct !{!29, !"air-alias-scopes(divshapes)"}
!30 = !{!31, !32}
!31 = distinct !{!31, !29, !"air-alias-scope-arg(1)"}
!32 = distinct !{!32, !29, !"air-alias-scope-arg(2)"}
!33 = !{!32}
!34 = !{!28, !31}
!35 = !{!31}
!36 = !{!28, !32}
