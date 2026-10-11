; ModuleID = 'n.fix.ll'
source_filename = "/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_device_pointer_chain_phi.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

%struct.Node = type { ptr addrspace(1), i32 }

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define void @kernel_device_pointer_chain_phi(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(2) readonly align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %2, i32 %3) local_unnamed_addr #0 {
  %5 = load i32, ptr addrspace(2) %2, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %15

7:                                                ; preds = %15, %4
  %8 = phi i32 [ 0, %4 ], [ %21, %15 ]
  %9 = phi ptr addrspace(1) [ %0, %4 ], [ %25, %15 ]
  %10 = getelementptr inbounds %struct.Node, ptr addrspace(1) %9, i64 0, i32 1
  %11 = load i32, ptr addrspace(1) %10, align 8, !tbaa !36
  %12 = add i32 %11, %8
  %13 = zext i32 %3 to i64
  %14 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %13
  store i32 %12, ptr addrspace(1) %14, align 4, !tbaa !26, !alias.scope !39, !noalias !40
  ret void

15:                                               ; preds = %15, %4
  %16 = phi ptr addrspace(1) [ %25, %15 ], [ %0, %4 ]
  %17 = phi i32 [ %26, %15 ], [ 0, %4 ]
  %18 = phi i32 [ %21, %15 ], [ 0, %4 ]
  %19 = getelementptr inbounds %struct.Node, ptr addrspace(1) %16, i64 0, i32 1
  %20 = load i32, ptr addrspace(1) %19, align 8, !tbaa !36
  %21 = add i32 %20, %18
  %22 = bitcast ptr addrspace(1) %16 to ptr addrspace(1)
  %23 = load ptr addrspace(1), ptr addrspace(1) %22, align 8, !tbaa !41
  %24 = icmp eq ptr addrspace(1) %23, null
  %25 = select i1 %24, ptr addrspace(1) %16, ptr addrspace(1) %23
  %26 = add nuw i32 %17, 1
  %27 = icmp eq i32 %26, %5
  br i1 %27, label %7, label %15, !llvm.loop !42
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!19, !20, !21}
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
!9 = !{ptr @kernel_device_pointer_chain_phi, !10, !11}
!10 = !{}
!11 = !{!12, !16, !17, !18}
!12 = !{i32 0, !"air.indirect_buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !13, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Node", !"air.arg_name", !"root"}
!13 = !{i32 0, i32 8, i32 0, !"uint", !"next", !"air.indirect_argument", !14, i32 8, i32 4, i32 0, !"uint", !"value", !"air.indirect_argument", !15}
!14 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"next"}
!15 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"value"}
!16 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!17 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"steps"}
!18 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 3, i32 0, i32 0}
!25 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_device_pointer_chain_phi.metal"}
!26 = !{!27, !27, i64 0}
!27 = !{!"int", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"air-alias-scope-arg(2)"}
!32 = distinct !{!32, !"air-alias-scopes(kernel_device_pointer_chain_phi)"}
!33 = !{!34, !35}
!34 = distinct !{!34, !32, !"air-alias-scope-arg(0)"}
!35 = distinct !{!35, !32, !"air-alias-scope-arg(1)"}
!36 = !{!37, !27, i64 8}
!37 = !{!"_ZTS4Node", !38, i64 0, !27, i64 8}
!38 = !{!"any pointer", !28, i64 0}
!39 = !{!35}
!40 = !{!34, !31}
!41 = !{!37, !38, i64 0}
!42 = distinct !{!42, !43}
!43 = !{!"llvm.loop.mustprogress"}
