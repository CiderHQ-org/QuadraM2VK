; ModuleID = 'kernel_uniform_member_helper_twice.air'
source_filename = "validation/fixtures/public/kernel_uniform_member_helper_twice.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Params = type { %struct.Coef, %struct.Coef, float }
%struct.Coef = type { float, float, float, float }

; Function Attrs: convergent mustprogress nofree nosync nounwind willreturn memory(argmem: write)
define void @uniform_member_helper_twice(ptr addrspace(2) noundef readonly align 4 captures(none) dereferenceable(36) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = getelementptr inbounds %struct.Params, ptr addrspace(2) %0, i64 0, i32 0
  %5 = tail call fast float @air.convert.f.f32.u.i32(i32 %2) #3
  %6 = tail call fast fastcc float @_ZL4polyRU11MTLconstantK4Coeff(ptr addrspace(2) noundef align 4 dereferenceable(16) %4, float noundef %5) #4
  %7 = and i32 %2, 1
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %13, label %9

9:                                                ; preds = %3
  %10 = fadd fast float %5, 1.000000e+00
  %11 = tail call fast fastcc float @_ZL4polyRU11MTLconstantK4Coeff(ptr addrspace(2) noundef align 4 dereferenceable(16) %4, float noundef %10) #4
  %12 = fadd fast float %11, %6
  br label %13

13:                                               ; preds = %9, %3
  %14 = phi float [ %12, %9 ], [ %6, %3 ]
  %15 = getelementptr inbounds %struct.Params, ptr addrspace(2) %0, i64 0, i32 2
  %16 = load float, ptr addrspace(2) %15, align 4, !tbaa !24, !alias.scope !30, !noalias !33
  %17 = fmul fast float %16, %14
  %18 = zext i32 %2 to i64
  %19 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %18
  store float %17, ptr addrspace(1) %19, align 4, !tbaa !35, !alias.scope !33, !noalias !30
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none)
define internal fastcc float @_ZL4polyRU11MTLconstantK4Coeff(ptr addrspace(2) noundef readonly align 4 captures(none) dereferenceable(16) %0, float noundef %1) unnamed_addr #1 {
  %3 = fcmp fast olt float %1, 0.000000e+00
  br i1 %3, label %4, label %7

4:                                                ; preds = %2
  %5 = getelementptr inbounds %struct.Coef, ptr addrspace(2) %0, i64 0, i32 0
  %6 = load float, ptr addrspace(2) %5, align 4, !tbaa !36
  br label %22

7:                                                ; preds = %2
  %8 = getelementptr inbounds %struct.Coef, ptr addrspace(2) %0, i64 0, i32 3
  %9 = load float, ptr addrspace(2) %8, align 4, !tbaa !37
  %10 = fmul fast float %9, %1
  %11 = getelementptr inbounds %struct.Coef, ptr addrspace(2) %0, i64 0, i32 2
  %12 = load float, ptr addrspace(2) %11, align 4, !tbaa !38
  %13 = fadd fast float %10, %12
  %14 = fmul fast float %13, %1
  %15 = getelementptr inbounds %struct.Coef, ptr addrspace(2) %0, i64 0, i32 1
  %16 = load float, ptr addrspace(2) %15, align 4, !tbaa !39
  %17 = fadd fast float %14, %16
  %18 = fmul fast float %17, %1
  %19 = getelementptr inbounds %struct.Coef, ptr addrspace(2) %0, i64 0, i32 0
  %20 = load float, ptr addrspace(2) %19, align 4, !tbaa !36
  %21 = fadd fast float %18, %20
  br label %22

22:                                               ; preds = %7, %4
  %23 = phi float [ %6, %4 ], [ %21, %7 ]
  ret float %23
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #2

attributes #0 = { convergent mustprogress nofree nosync nounwind willreturn memory(argmem: write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nounwind willreturn memory(none) }
attributes #4 = { nobuiltin "no-builtins" }

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
!9 = !{ptr @uniform_member_helper_twice, !10, !11}
!10 = !{}
!11 = !{!12, !15, !16}
!12 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 36, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !13, !"air.arg_type_size", i32 36, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"p"}
!13 = !{!"air.struct_type_info", !14, i32 0, i32 16, i32 0, !"Coef", !"a", !"air.struct_type_info", !14, i32 16, i32 16, i32 0, !"Coef", !"b", i32 32, i32 4, i32 0, !"float", !"scale"}
!14 = !{i32 0, i32 4, i32 0, !"float", !"c0", i32 4, i32 4, i32 0, !"float", !"c1", i32 8, i32 4, i32 0, !"float", !"c2", i32 12, i32 4, i32 0, !"float", !"c3"}
!15 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!16 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 4, i32 0, i32 0}
!23 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_uniform_member_helper_twice.metal"}
!24 = !{!25, !27, i64 32}
!25 = !{!"_ZTS6Params", !26, i64 0, !26, i64 16, !27, i64 32}
!26 = !{!"_ZTS4Coef", !27, i64 0, !27, i64 4, !27, i64 8, !27, i64 12}
!27 = !{!"float", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"air-alias-scope-arg(0)"}
!32 = distinct !{!32, !"air-alias-scopes(uniform_member_helper_twice)"}
!33 = !{!34}
!34 = distinct !{!34, !32, !"air-alias-scope-arg(1)"}
!35 = !{!27, !27, i64 0}
!36 = !{!26, !27, i64 0}
!37 = !{!26, !27, i64 12}
!38 = !{!26, !27, i64 8}
!39 = !{!26, !27, i64 4}
