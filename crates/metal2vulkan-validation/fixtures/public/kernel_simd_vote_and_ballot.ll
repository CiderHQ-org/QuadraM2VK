; ModuleID = '/tmp/b3/f.bc'
source_filename = "validation/fixtures/public/kernel_simd_vote_and_ballot.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_simd_vote_and_ballot(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = load i32, ptr addrspace(1) %1, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %5 = and i32 %2, 31
  %6 = shl nuw i32 1, %5
  %7 = and i32 %4, %6
  %8 = icmp ne i32 %7, 0
  %9 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 1
  %10 = load i32, ptr addrspace(1) %9, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %11 = and i32 %10, %6
  %12 = icmp ne i32 %11, 0
  %13 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 2
  %14 = load i32, ptr addrspace(1) %13, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %15 = and i32 %14, %6
  %16 = icmp ne i32 %15, 0
  %17 = tail call i64 @air.simd_ballot.i64(i1 %8) #2
  %18 = tail call i64 @air.simd_ballot.i64(i1 %12) #2
  %19 = tail call i64 @air.simd_ballot.i64(i1 %16) #2
  %20 = shl i32 %2, 3
  %21 = zext i32 %20 to i64
  %22 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %21
  %23 = trunc i64 %17 to i32
  store i32 %23, ptr addrspace(1) %22, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %24 = lshr i64 %17, 32
  %25 = trunc i64 %24 to i32
  %26 = getelementptr inbounds i32, ptr addrspace(1) %22, i64 1
  store i32 %25, ptr addrspace(1) %26, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %27 = trunc i64 %18 to i32
  %28 = getelementptr inbounds i32, ptr addrspace(1) %22, i64 2
  store i32 %27, ptr addrspace(1) %28, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %29 = trunc i64 %19 to i32
  %30 = getelementptr inbounds i32, ptr addrspace(1) %22, i64 3
  store i32 %29, ptr addrspace(1) %30, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %31 = tail call i1 @air.simd_any(i1 %8) #2
  %32 = zext i1 %31 to i32
  %33 = tail call i1 @air.simd_all(i1 %8) #2
  %34 = select i1 %33, i32 2, i32 0
  %35 = or i32 %34, %32
  %36 = tail call i1 @air.simd_any(i1 %12) #2
  %37 = select i1 %36, i32 4, i32 0
  %38 = or i32 %35, %37
  %39 = tail call i1 @air.simd_all(i1 %12) #2
  %40 = select i1 %39, i32 8, i32 0
  %41 = or i32 %38, %40
  %42 = tail call i1 @air.simd_any(i1 %16) #2
  %43 = select i1 %42, i32 16, i32 0
  %44 = or i32 %41, %43
  %45 = tail call i1 @air.simd_all(i1 %16) #2
  %46 = select i1 %45, i32 32, i32 0
  %47 = or i32 %44, %46
  %48 = tail call i1 @air.simd_is_first() #2
  %49 = select i1 %48, i32 64, i32 0
  %50 = or i32 %47, %49
  %51 = getelementptr inbounds i32, ptr addrspace(1) %22, i64 4
  store i32 %50, ptr addrspace(1) %51, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %52 = add i32 %2, 100
  %53 = tail call i32 @air.simd_broadcast_first.u.i32(i32 %52) #2
  %54 = getelementptr inbounds i32, ptr addrspace(1) %22, i64 5
  store i32 %53, ptr addrspace(1) %54, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %55 = getelementptr inbounds i32, ptr addrspace(1) %22, i64 6
  store i32 %2, ptr addrspace(1) %55, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %56 = lshr i64 %18, 32
  %57 = trunc i64 %56 to i32
  %58 = getelementptr inbounds i32, ptr addrspace(1) %22, i64 7
  store i32 %57, ptr addrspace(1) %58, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare i64 @air.simd_ballot.i64(i1) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i1 @air.simd_any(i1) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i1 @air.simd_all(i1) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i1 @air.simd_is_first() local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.simd_broadcast_first.u.i32(i32) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent nounwind willreturn }

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
!9 = !{ptr @kernel_simd_vote_and_ballot, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"masks"}
!14 = !{i32 2, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 3, i32 2, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_simd_vote_and_ballot.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_simd_vote_and_ballot)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(0)"}
