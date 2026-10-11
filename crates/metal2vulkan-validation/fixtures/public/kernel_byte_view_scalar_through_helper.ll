; Hand edit (DESIGN D149): the `llvm.lifetime.*` calls and declarations take the two
; arguments LLVM 21's `llvm-dis` writes (`i64 <alloca size in bytes>, ptr`), which LLVM 22+
; auto-upgrades by dropping the size, so every gate's `llvm-as` accepts the text.
; ModuleID = '/tmp/mkfix2/fx2.bc'
source_filename = "validation/fixtures/public/kernel_byte_view_scalar_through_helper.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.record_view = type { ptr addrspace(1) }

; Function Attrs: convergent mustprogress nofree nosync nounwind willreturn
define void @kernel_byte_view_scalar_through_helper(ptr addrspace(1) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = alloca %struct.record_view, align 8
  %5 = bitcast ptr %4 to ptr
  call void @llvm.lifetime.start.p0(i64 8, ptr %4)
  %6 = getelementptr inbounds %struct.record_view, ptr %4, i64 0, i32 0
  store ptr addrspace(1) %0, ptr %6, align 8, !tbaa !22
  %7 = shl i32 %2, 4
  %8 = call fast fastcc <4 x float> @_ZL11read_recordRK11record_viewj(ptr noundef nonnull align 8 dereferenceable(8) %4, i32 noundef %7) #3
  %9 = zext i32 %2 to i64
  %10 = getelementptr inbounds <4 x float>, ptr addrspace(1) %1, i64 %9
  store <4 x float> %8, ptr addrspace(1) %10, align 16, !tbaa !27, !alias.scope !28, !noalias !31
  call void @llvm.lifetime.end.p0(i64 8, ptr %4)
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read)
define internal fastcc <4 x float> @_ZL11read_recordRK11record_viewj(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %0, i32 noundef %1) unnamed_addr #1 {
  %3 = getelementptr inbounds %struct.record_view, ptr %0, i64 0, i32 0
  %4 = load ptr addrspace(1), ptr %3, align 8, !tbaa !22
  %5 = zext i32 %1 to i64
  %6 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %5
  %7 = bitcast ptr addrspace(1) %6 to ptr addrspace(1)
  %8 = load float, ptr addrspace(1) %7, align 4
  %9 = insertelement <4 x float> undef, float %8, i64 0
  %10 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 4
  %11 = bitcast ptr addrspace(1) %10 to ptr addrspace(1)
  %12 = load float, ptr addrspace(1) %11, align 4
  %13 = insertelement <4 x float> %9, float %12, i64 1
  %14 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 8
  %15 = bitcast ptr addrspace(1) %14 to ptr addrspace(1)
  %16 = load float, ptr addrspace(1) %15, align 4
  %17 = insertelement <4 x float> %13, float %16, i64 2
  %18 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 12
  %19 = bitcast ptr addrspace(1) %18 to ptr addrspace(1)
  %20 = load float, ptr addrspace(1) %19, align 4, !tbaa !33
  %21 = insertelement <4 x float> %17, float %20, i64 3
  ret <4 x float> %21
}

declare void @llvm.lifetime.start.i64(i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #2

declare void @llvm.lifetime.end.i64(i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #2

attributes #0 = { convergent mustprogress nofree nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nobuiltin "no-builtins" }

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
!9 = !{ptr @kernel_byte_view_scalar_through_helper, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_byte_view_scalar_through_helper.metal"}
!22 = !{!23, !24, i64 0}
!23 = !{!"_ZTS11record_view", !24, i64 0}
!24 = !{!"any pointer", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!25, !25, i64 0}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(1)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_byte_view_scalar_through_helper)"}
!31 = !{!32}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(0)"}
!33 = !{!34, !34, i64 0}
!34 = !{!"float", !25, i64 0}
