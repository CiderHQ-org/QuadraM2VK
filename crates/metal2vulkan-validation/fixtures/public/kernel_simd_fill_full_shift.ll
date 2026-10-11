; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = '/tmp/alias/simd.bc'
source_filename = "validation/fixtures/public/kernel_simd_fill_full_shift.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_simd_fill_full_shift(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add i32 %1, 100
  %4 = add i32 %1, 200
  %5 = tail call i32 @air.simd_shuffle_and_fill_up.u.i32(i32 %3, i32 %4, i16 0, i16 32) #2
  %6 = shl i32 %1, 3
  %7 = zext i32 %6 to i64
  %8 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %7
  store i32 %5, ptr addrspace(1) %8, align 4, !tbaa !21, !alias.scope !25
  %9 = tail call i32 @air.simd_shuffle_and_fill_up.u.i32(i32 %3, i32 %4, i16 1, i16 32) #2
  %10 = or i32 %6, 1
  %11 = zext i32 %10 to i64
  %12 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %11
  store i32 %9, ptr addrspace(1) %12, align 4, !tbaa !21, !alias.scope !25
  %13 = tail call i32 @air.simd_shuffle_and_fill_up.u.i32(i32 %3, i32 %4, i16 32, i16 32) #2
  %14 = or i32 %6, 2
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %15
  store i32 %13, ptr addrspace(1) %16, align 4, !tbaa !21, !alias.scope !25
  %17 = tail call i32 @air.simd_shuffle_and_fill_down.u.i32(i32 %3, i32 %4, i16 32, i16 32) #2
  %18 = or i32 %6, 3
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %19
  store i32 %17, ptr addrspace(1) %20, align 4, !tbaa !21, !alias.scope !25
  %21 = tail call i32 @air.simd_shuffle_and_fill_up.u.i32(i32 %3, i32 %4, i16 8, i16 8) #2
  %22 = or i32 %6, 4
  %23 = zext i32 %22 to i64
  %24 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %23
  store i32 %21, ptr addrspace(1) %24, align 4, !tbaa !21, !alias.scope !25
  %25 = tail call i32 @air.simd_shuffle_and_fill_down.u.i32(i32 %3, i32 %4, i16 8, i16 8) #2
  %26 = or i32 %6, 5
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %27
  store i32 %25, ptr addrspace(1) %28, align 4, !tbaa !21, !alias.scope !25
  %29 = tail call i32 @air.simd_shuffle_and_fill_up.u.i32(i32 %3, i32 %4, i16 3, i16 8) #2
  %30 = or i32 %6, 6
  %31 = zext i32 %30 to i64
  %32 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %31
  store i32 %29, ptr addrspace(1) %32, align 4, !tbaa !21, !alias.scope !25
  %33 = tail call i32 @air.simd_shuffle_and_fill_down.u.i32(i32 %3, i32 %4, i16 3, i16 8) #2
  %34 = or i32 %6, 7
  %35 = zext i32 %34 to i64
  %36 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %35
  store i32 %33, ptr addrspace(1) %36, align 4, !tbaa !21, !alias.scope !25
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.simd_shuffle_and_fill_up.u.i32(i32, i32, i16, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.simd_shuffle_and_fill_down.u.i32(i32, i32, i16, i16) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!14, !15, !16}
!llvm.ident = !{!17}
!air.version = !{!18}
!air.language_version = !{!19}
!air.source_file_name = !{!20}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_simd_fill_full_shift, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_simd_fill_full_shift.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_simd_fill_full_shift)"}
