; Generated from kernel_absent_texture_reads_zero.metal by
;   xcrun metal -std=metal3.0 -S -emit-llvm, retripled to spirv-unknown-vulkan1.2, opt -S.
;
; Two edits, and both spell the SPECIALIZED variant the corpus is in. The frontend emits the
; UNspecialized module, where a gated texture still carries a literal `air.location_index`; a module
; built with `provide_absent` off carries a static-initialized global there instead, because a slot
; behind a gate is a running sum -- see `meta::variant_texture_slot` and the corpus's own
; `@_ZL33__metal_implicit_attr_int_expr_*` globals. So:
;
;   * `@_ZL33__metal_implicit_attr_int_expr_0`/`_1` are declared, and stored 0 in `air.static_init`;
;   * the two gated textures name them in `air.location_index` instead of a literal.
;
; That is the whole difference from the frontend's output, and it is what makes this module describe
; the variant its .metal is RUN as: `provide_absent = false`, so `absent_source` and `absent_sink`
; are resources the pipeline does not provide.

; ModuleID = 'kernel_absent_texture_reads_zero'
source_filename = "kernel_absent_texture_reads_zero.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

@_Z14provide_absent.MTL_FC_INIT_0_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i8 0, align 1
@_ZL33__metal_implicit_attr_int_expr_0 = internal addrspace(2) global i32 0, align 4
@_ZL33__metal_implicit_attr_int_expr_1 = internal addrspace(2) global i32 0, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_kernel_absent_texture_reads_zero, ptr null }]
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(2) @__metal_implicit_fc_pred_0 to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nounwind willreturn
define void @absent_texture_reads_zero(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, <2 x i32> %4) local_unnamed_addr #0 {
  %6 = extractelement <2 x i32> %4, i64 0
  %7 = icmp eq i32 %6, 0
  %8 = extractelement <2 x i32> %4, i64 1
  %9 = icmp eq i32 %8, 0
  %10 = select i1 %7, i1 %9, i1 false
  br i1 %10, label %11, label %15

11:                                               ; preds = %5
  %12 = tail call ptr addrspace(2) @air.get_read_sampler() #6
  %13 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %2, ptr addrspace(2) %12, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1) #7
  %14 = extractvalue { <4 x float>, i8 } %13, 0
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none) %3, <2 x i32> %4, <4 x float> %14, i32 0, i32 2) #8, !alias.scope !26
  br label %15

15:                                               ; preds = %11, %5
  %16 = icmp eq i32 %6, 1
  %17 = select i1 %16, i1 %9, i1 false
  br i1 %17, label %18, label %22

18:                                               ; preds = %15
  %19 = tail call ptr addrspace(2) @air.get_read_sampler() #6
  %20 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %19, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1) #7
  %21 = extractvalue { <4 x float>, i8 } %20, 0
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none) %3, <2 x i32> %4, <4 x float> %21, i32 0, i32 2) #8, !alias.scope !26
  br label %22

22:                                               ; preds = %18, %15
  %23 = icmp eq i32 %8, 1
  br i1 %23, label %24, label %25

24:                                               ; preds = %22
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none) %3, <2 x i32> %4, <4 x float> splat (float 4.200000e+01), i32 0, i32 2) #8, !alias.scope !26
  br label %25

25:                                               ; preds = %24, %22
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none) %1, <2 x i32> %4, <4 x float> splat (float -1.000000e+00), i32 0, i32 2) #8, !alias.scope !26
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i8 @air.normalize_function_constant_predicate.i8(i8) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: read)
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none), <2 x i32>, <4 x float>, i32, i32) local_unnamed_addr #4

; Function Attrs: mustprogress nofree nosync nounwind willreturn
define internal void @_GLOBAL__sub_I_kernel_absent_texture_reads_zero() #5 section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @_Z14provide_absent.MTL_FC_INIT_0_b, align 1, !tbaa !29, !range !33
  %2 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1) #9
  store i8 %2, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 1
  store i32 0, ptr addrspace(2) @_ZL33__metal_implicit_attr_int_expr_0, align 4
  store i32 0, ptr addrspace(2) @_ZL33__metal_implicit_attr_int_expr_1, align 4
  ret void
}

attributes #0 = { mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #3 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nofree nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #6 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #7 = { nounwind willreturn memory(argmem: read) }
attributes #8 = { nounwind willreturn memory(argmem: readwrite) }
attributes #9 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!18, !19, !20}
!air.function_constants = !{!21}
!llvm.ident = !{!22}
!air.version = !{!23}
!air.language_version = !{!24}
!air.source_file_name = !{!25}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @absent_texture_reads_zero, !10, !11}
!10 = !{}
!11 = !{!12, !14, !15, !16, !17}
!12 = !{i32 0, !"air.function_constant", !13, !"air.texture", !"air.location_index", ptr addrspace(2) @_ZL33__metal_implicit_attr_int_expr_0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"absent_source"}
!13 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"provide_absent"}
!14 = !{i32 1, !"air.function_constant", !13, !"air.texture", !"air.location_index", ptr addrspace(2) @_ZL33__metal_implicit_attr_int_expr_1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"absent_sink"}
!15 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"present"}
!16 = !{i32 3, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"dest"}
!17 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{ptr addrspace(2) @_Z14provide_absent.MTL_FC_INIT_0_b, !"bool", !"provide_absent", i32 0}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 3, i32 0, i32 0}
!25 = !{!"kernel_absent_texture_reads_zero.metal"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-textures"}
!28 = distinct !{!28, !"air-alias-scopes(absent_texture_reads_zero)"}
!29 = !{!30, !30, i64 0}
!30 = !{!"bool", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C++ TBAA"}
!33 = !{i8 0, i8 2}
