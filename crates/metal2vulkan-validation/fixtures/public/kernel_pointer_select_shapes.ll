; ModuleID = 'kernel_pointer_select_shapes.ll'
source_filename = "kernel_pointer_select_shapes.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind willreturn
define void @ptrsel(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %3, i32 %4) local_unnamed_addr #0 {
  %6 = and i32 %4, 1
  %7 = icmp ne i32 %6, 0
  %8 = select i1 %7, ptr addrspace(1) %0, ptr addrspace(1) %1
  %9 = tail call fast fastcc float @_ZL7read_atPU9MTLdeviceKfj(ptr addrspace(1) %8, i32 %4) #2, !alias.scope !24, !noalias !28
  %10 = zext i32 %4 to i64
  %11 = getelementptr inbounds float, ptr addrspace(1) %3, i64 %10
  store float %9, ptr addrspace(1) %11, align 4, !tbaa !31, !alias.scope !35, !noalias !36
  %12 = add i32 %4, 3
  %13 = and i32 %12, 7
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds float, ptr addrspace(1) %8, i64 %14
  %16 = load float, ptr addrspace(1) %15, align 4, !tbaa !31, !alias.scope !24, !noalias !28
  %17 = add i32 %4, 8
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds float, ptr addrspace(1) %3, i64 %18
  store float %16, ptr addrspace(1) %19, align 4, !tbaa !31, !alias.scope !35, !noalias !36
  br i1 %7, label %20, label %24

20:                                               ; preds = %5
  %21 = and i32 %4, 3
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %22
  br label %29

24:                                               ; preds = %5
  %25 = lshr i32 %4, 1
  %26 = and i32 %25, 3
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %27
  br label %29

29:                                               ; preds = %24, %20
  %30 = phi ptr addrspace(1) [ %23, %20 ], [ %28, %24 ]
  %31 = load float, ptr addrspace(1) %30, align 4, !tbaa !31, !alias.scope !24, !noalias !28
  %32 = add i32 %4, 16
  %33 = zext i32 %32 to i64
  %34 = getelementptr inbounds float, ptr addrspace(1) %3, i64 %33
  store float %31, ptr addrspace(1) %34, align 4, !tbaa !31, !alias.scope !35, !noalias !36
  %35 = getelementptr inbounds float, ptr addrspace(1) %30, i64 1
  %36 = load float, ptr addrspace(1) %35, align 4, !tbaa !31, !alias.scope !24, !noalias !28
  %37 = tail call fast fastcc float @_ZL7read_atPU9MTLdeviceKfj(ptr addrspace(1) %30, i32 2) #2, !alias.scope !24, !noalias !28
  %38 = fadd fast float %37, %36
  %39 = add i32 %4, 24
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds float, ptr addrspace(1) %3, i64 %40
  store float %38, ptr addrspace(1) %41, align 4, !tbaa !31, !alias.scope !35, !noalias !36
  %42 = shl nuw nsw i32 %6, 4
  %43 = xor i32 %42, 16
  %44 = zext i32 %43 to i64
  %45 = getelementptr i8, ptr addrspace(1) %2, i64 %44
  %46 = shl i32 %4, 2
  %47 = and i32 %46, 12
  %48 = tail call fastcc i32 @_ZL9read_wordPU9MTLdeviceKhj(ptr addrspace(1) %45, i32 %47) #2, !alias.scope !37, !noalias !38
  %49 = add i32 %4, 32
  %50 = zext i32 %49 to i64
  %51 = getelementptr inbounds float, ptr addrspace(1) %3, i64 %50
  %52 = bitcast ptr addrspace(1) %51 to ptr addrspace(1)
  store i32 %48, ptr addrspace(1) %52, align 4, !tbaa !31, !alias.scope !35, !noalias !36
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read)
define internal fastcc float @_ZL7read_atPU9MTLdeviceKfj(ptr addrspace(1) readonly captures(none) %0, i32 %1) unnamed_addr #1 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %3
  %5 = load float, ptr addrspace(1) %4, align 4, !tbaa !31
  ret float %5
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read)
define internal fastcc i32 @_ZL9read_wordPU9MTLdeviceKhj(ptr addrspace(1) readonly captures(none) %0, i32 %1) unnamed_addr #1 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds i8, ptr addrspace(1) %0, i64 %3
  %5 = bitcast ptr addrspace(1) %4 to ptr addrspace(1)
  %6 = load i32, ptr addrspace(1) %5, align 4, !tbaa !39
  ret i32 %6
}

attributes #0 = { convergent mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { nobuiltin "no-builtins" }

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
!9 = !{ptr @ptrsel, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"raw"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!16 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 3, i32 0, i32 0}
!23 = !{!"/private/tmp/ptrcase/kernel_pointer_select_shapes.metal"}
!24 = !{!25, !27}
!25 = distinct !{!25, !26, !"air-alias-scope-arg(0)"}
!26 = distinct !{!26, !"air-alias-scopes(ptrsel)"}
!27 = distinct !{!27, !26, !"air-alias-scope-arg(1)"}
!28 = !{!29, !30}
!29 = distinct !{!29, !26, !"air-alias-scope-arg(2)"}
!30 = distinct !{!30, !26, !"air-alias-scope-arg(3)"}
!31 = !{!32, !32, i64 0}
!32 = !{!"float", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C++ TBAA"}
!35 = !{!30}
!36 = !{!25, !27, !29}
!37 = !{!29}
!38 = !{!25, !27, !30}
!39 = !{!40, !40, i64 0}
!40 = !{!"int", !33, i64 0}
