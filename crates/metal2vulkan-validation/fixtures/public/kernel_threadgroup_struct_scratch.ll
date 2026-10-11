; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_threadgroup_struct_scratch.bc'
source_filename = "validation/fixtures/public/kernel_threadgroup_struct_scratch.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Wide = type { [5 x <4 x i32>] }

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_threadgroup_struct_scratch(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(3) noundef captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = mul i32 %2, 3
  %5 = add i32 %4, 1
  %6 = insertelement <4 x i32> undef, i32 %5, i64 0
  %7 = mul i32 %2, 5
  %8 = add i32 %7, 2
  %9 = insertelement <4 x i32> %6, i32 %8, i64 1
  %10 = mul i32 %2, 7
  %11 = add i32 %10, 3
  %12 = insertelement <4 x i32> %9, i32 %11, i64 2
  %13 = mul i32 %2, 11
  %14 = add i32 %13, 4
  %15 = insertelement <4 x i32> %12, i32 %14, i64 3
  %16 = zext i32 %2 to i64
  %17 = getelementptr inbounds %struct.Wide, ptr addrspace(3) %1, i64 %16, i32 0, i64 0
  store <4 x i32> %15, ptr addrspace(3) %17, align 16, !tbaa !23, !alias.scope !26, !noalias !29
  %18 = mul i32 %2, 13
  %19 = add i32 %18, 5
  %20 = insertelement <4 x i32> <i32 poison, i32 0, i32 0, i32 0>, i32 %19, i64 0
  %21 = getelementptr inbounds %struct.Wide, ptr addrspace(3) %1, i64 %16, i32 0, i64 4
  store <4 x i32> %20, ptr addrspace(3) %21, align 16, !tbaa !23, !alias.scope !26, !noalias !29
  tail call void @air.wg.barrier(i32 2, i32 1) #2
  %22 = add i32 %2, 1
  %23 = and i32 %22, 31
  %24 = add i32 %2, 2
  %25 = and i32 %24, 31
  %26 = zext i32 %23 to i64
  %27 = getelementptr inbounds %struct.Wide, ptr addrspace(3) %1, i64 %26, i32 0, i64 0
  %28 = load <4 x i32>, ptr addrspace(3) %27, align 16, !alias.scope !26, !noalias !29
  %29 = extractelement <4 x i32> %28, i64 0
  %30 = zext i32 %25 to i64
  %31 = getelementptr inbounds %struct.Wide, ptr addrspace(3) %1, i64 %30, i32 0, i64 0
  %32 = load <4 x i32>, ptr addrspace(3) %31, align 16, !alias.scope !26, !noalias !29
  %33 = extractelement <4 x i32> %32, i64 3
  %34 = add i32 %33, %29
  %35 = getelementptr inbounds %struct.Wide, ptr addrspace(3) %1, i64 %26, i32 0, i64 4
  %36 = load <4 x i32>, ptr addrspace(3) %35, align 16, !alias.scope !26, !noalias !29
  %37 = extractelement <4 x i32> %36, i64 0
  %38 = add i32 %34, %37
  %39 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %16
  store i32 %38, ptr addrspace(1) %39, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent nounwind willreturn }

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
!9 = !{ptr @kernel_threadgroup_struct_scratch, !10, !11}
!10 = !{}
!11 = !{!12, !13, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !14, !"air.arg_type_size", i32 80, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"Wide", !"air.arg_name", !"scratch"}
!14 = !{i32 0, i32 16, i32 5, !"uint4", !"v"}
!15 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lid"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/private/tmp/tgf/k.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_threadgroup_struct_scratch)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(0)"}
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !24, i64 0}
