; ModuleID = 'kernel_function_constant_gate_fold.ll'
source_filename = "kernel_function_constant_gate_fold.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

@_ZL4mode = internal unnamed_addr addrspace(2) global i32 undef, align 4
@_Z4mode.MTL_FC_INIT_0_i = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_kernel_function_constant_gate_fold.metal, ptr null }]

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @fcfold(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, i32 %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %4
  %6 = load float, ptr addrspace(1) %5, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %7 = fadd fast float %6, 1.000000e+00
  %8 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %4
  store float %7, ptr addrspace(1) %8, align 4, !tbaa !23, !alias.scope !30, !noalias !27
  %9 = load i32, ptr addrspace(2) @_ZL4mode, align 4, !tbaa !32
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %3
  %12 = fmul fast float %6, 1.000000e+03
  br label %18

13:                                               ; preds = %3
  %14 = add i32 %2, 8
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %15
  %17 = fmul fast float %6, 1.000000e+03
  store float %17, ptr addrspace(1) %16, align 4, !tbaa !23, !alias.scope !30, !noalias !27
  br label %18

18:                                               ; preds = %13, %11
  %19 = phi float [ %12, %11 ], [ %17, %13 ]
  %20 = icmp sgt i32 %9, 0
  %21 = fadd fast float %6, -1.000000e+00
  %22 = select fast i1 %20, float %19, float %21
  %23 = add i32 %2, 16
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %24
  store float %22, ptr addrspace(1) %25, align 4, !tbaa !23, !alias.scope !30, !noalias !27
  switch i32 %9, label %30 [
    i32 1, label %31
    i32 2, label %26
    i32 3, label %28
  ]

26:                                               ; preds = %18
  %27 = fmul fast float %6, 2.000000e+03
  br label %31

28:                                               ; preds = %18
  %29 = fmul fast float %6, 3.000000e+03
  br label %31

30:                                               ; preds = %18
  br label %31

31:                                               ; preds = %30, %28, %26, %18
  %32 = phi float [ %27, %26 ], [ %29, %28 ], [ %19, %18 ], [ %6, %30 ]
  %33 = add i32 %2, 24
  %34 = zext i32 %33 to i64
  %35 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %34
  store float %32, ptr addrspace(1) %35, align 4, !tbaa !23, !alias.scope !30, !noalias !27
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write)
define internal void @_GLOBAL__sub_I_kernel_function_constant_gate_fold.metal() #1 section "air.static_init" {
  %1 = load i32, ptr addrspace(2) @_Z4mode.MTL_FC_INIT_0_i, align 4, !tbaa !32
  store i32 %1, ptr addrspace(2) @_ZL4mode, align 4, !tbaa !32
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!air.function_constants = !{!18}
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
!9 = !{ptr @fcfold, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{ptr addrspace(2) @_Z4mode.MTL_FC_INIT_0_i, !"int", !"mode", i32 0}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 3, i32 0, i32 0}
!22 = !{!"/private/tmp/fc2/kernel_function_constant_gate_fold.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"float", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(0)"}
!29 = distinct !{!29, !"air-alias-scopes(fcfold)"}
!30 = !{!31}
!31 = distinct !{!31, !29, !"air-alias-scope-arg(1)"}
!32 = !{!33, !33, i64 0}
!33 = !{!"int", !25, i64 0}
