; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'sgb.bc'
source_filename = "validation/fixtures/public/kernel_simdgroup_matrix_block.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_simdgroup_matrix_block(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %4, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %5, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %6, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %7, i32 noundef %8) local_unnamed_addr #0 {
  %10 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) readonly captures(none) %0, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #4
  %11 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) readonly captures(none) %1, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #4
  %12 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) readonly captures(none) %2, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #4
  %13 = tail call fast <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1) readonly captures(none) %3, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #4
  %14 = tail call fast <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1) readonly captures(none) %4, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #4
  %15 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float 1.000000e+00) #5
  %16 = tail call fast <64 x half> @air.simdgroup_matrix_8x8_init_diag.v64f16.f16(half 0xH4000) #5
  %17 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %10, <64 x float> %11, <64 x float> %12) #5
  %18 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f32.v64f32(<64 x half> %13, <64 x float> %15, <64 x float> %12) #5
  %19 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %10, <64 x float> %15, <64 x float> %12) #5
  %20 = tail call fast <64 x half> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f16.v64f16.v64f16(<64 x half> %13, <64 x half> %16, <64 x half> %14) #5
  %21 = tail call fast <64 x half> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f32.v64f16.v64f32(<64 x float> %10, <64 x half> %16, <64 x float> %12) #5
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %17, ptr addrspace(1) writeonly captures(none) %6, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #6
  %22 = getelementptr inbounds float, ptr addrspace(1) %6, i64 64
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %18, ptr addrspace(1) writeonly captures(none) %22, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #6
  %23 = getelementptr inbounds float, ptr addrspace(1) %6, i64 128
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %19, ptr addrspace(1) writeonly captures(none) %23, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #6
  tail call void @air.simdgroup_matrix_8x8_store.v64f16.p1f16(<64 x half> %20, ptr addrspace(1) writeonly captures(none) %7, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #6
  %24 = getelementptr inbounds half, ptr addrspace(1) %7, i64 64
  tail call void @air.simdgroup_matrix_8x8_store.v64f16.p1f16(<64 x half> %21, ptr addrspace(1) writeonly captures(none) %24, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #6
  tail call void @air.wg.barrier(i32 1, i32 1) #5
  %25 = zext i32 %8 to i64
  %26 = getelementptr inbounds float, ptr addrspace(1) %6, i64 %25
  %27 = bitcast ptr addrspace(1) %26 to ptr addrspace(1)
  %28 = load i32, ptr addrspace(1) %27, align 4, !tbaa !28, !alias.scope !32, !noalias !35
  %29 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %25
  store i32 %28, ptr addrspace(1) %29, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %30 = add i32 %8, 32
  %31 = zext i32 %30 to i64
  %32 = getelementptr inbounds float, ptr addrspace(1) %6, i64 %31
  %33 = bitcast ptr addrspace(1) %32 to ptr addrspace(1)
  %34 = load i32, ptr addrspace(1) %33, align 4, !tbaa !28, !alias.scope !32, !noalias !35
  %35 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %31
  store i32 %34, ptr addrspace(1) %35, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %36 = add i32 %8, 64
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds float, ptr addrspace(1) %6, i64 %37
  %39 = bitcast ptr addrspace(1) %38 to ptr addrspace(1)
  %40 = load i32, ptr addrspace(1) %39, align 4, !tbaa !28, !alias.scope !32, !noalias !35
  %41 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %37
  store i32 %40, ptr addrspace(1) %41, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %42 = add i32 %8, 96
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds float, ptr addrspace(1) %6, i64 %43
  %45 = bitcast ptr addrspace(1) %44 to ptr addrspace(1)
  %46 = load i32, ptr addrspace(1) %45, align 4, !tbaa !28, !alias.scope !32, !noalias !35
  %47 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %43
  store i32 %46, ptr addrspace(1) %47, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %48 = add i32 %8, 128
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds float, ptr addrspace(1) %6, i64 %49
  %51 = bitcast ptr addrspace(1) %50 to ptr addrspace(1)
  %52 = load i32, ptr addrspace(1) %51, align 4, !tbaa !28, !alias.scope !32, !noalias !35
  %53 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %49
  store i32 %52, ptr addrspace(1) %53, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %54 = add i32 %8, 160
  %55 = zext i32 %54 to i64
  %56 = getelementptr inbounds float, ptr addrspace(1) %6, i64 %55
  %57 = bitcast ptr addrspace(1) %56 to ptr addrspace(1)
  %58 = load i32, ptr addrspace(1) %57, align 4, !tbaa !28, !alias.scope !32, !noalias !35
  %59 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %55
  store i32 %58, ptr addrspace(1) %59, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %60 = getelementptr inbounds half, ptr addrspace(1) %7, i64 %25
  %61 = bitcast ptr addrspace(1) %60 to ptr addrspace(1)
  %62 = load i16, ptr addrspace(1) %61, align 2, !tbaa !47, !alias.scope !49, !noalias !50
  %63 = zext i16 %62 to i32
  %64 = add i32 %8, 192
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %65
  store i32 %63, ptr addrspace(1) %66, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %67 = getelementptr inbounds half, ptr addrspace(1) %7, i64 %31
  %68 = bitcast ptr addrspace(1) %67 to ptr addrspace(1)
  %69 = load i16, ptr addrspace(1) %68, align 2, !tbaa !47, !alias.scope !49, !noalias !50
  %70 = zext i16 %69 to i32
  %71 = add i32 %8, 224
  %72 = zext i32 %71 to i64
  %73 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %72
  store i32 %70, ptr addrspace(1) %73, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %74 = getelementptr inbounds half, ptr addrspace(1) %7, i64 %37
  %75 = bitcast ptr addrspace(1) %74 to ptr addrspace(1)
  %76 = load i16, ptr addrspace(1) %75, align 2, !tbaa !47, !alias.scope !49, !noalias !50
  %77 = zext i16 %76 to i32
  %78 = add i32 %8, 256
  %79 = zext i32 %78 to i64
  %80 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %79
  store i32 %77, ptr addrspace(1) %80, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %81 = getelementptr inbounds half, ptr addrspace(1) %7, i64 %43
  %82 = bitcast ptr addrspace(1) %81 to ptr addrspace(1)
  %83 = load i16, ptr addrspace(1) %82, align 2, !tbaa !47, !alias.scope !49, !noalias !50
  %84 = zext i16 %83 to i32
  %85 = add i32 %8, 288
  %86 = zext i32 %85 to i64
  %87 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %86
  store i32 %84, ptr addrspace(1) %87, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) readonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1) readonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <64 x half> @air.simdgroup_matrix_8x8_init_diag.v64f16.f16(half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f32.v64f32(<64 x half>, <64 x float>, <64 x float>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <64 x half> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f16.v64f16.v64f16(<64 x half>, <64 x half>, <64 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <64 x half> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f32.v64f16.v64f32(<64 x float>, <64 x half>, <64 x float>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn memory(write)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float>, ptr addrspace(1) writeonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn memory(write)
declare void @air.simdgroup_matrix_8x8_store.v64f16.p1f16(<64 x half>, ptr addrspace(1) writeonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #3

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="2048" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(read) }
attributes #3 = { convergent mustprogress nounwind willreturn memory(write) }
attributes #4 = { convergent nounwind willreturn memory(read) }
attributes #5 = { convergent nounwind willreturn }
attributes #6 = { convergent nounwind willreturn memory(write) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!21, !22, !23}
!llvm.ident = !{!24}
!air.version = !{!25}
!air.language_version = !{!26}
!air.source_file_name = !{!27}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_simdgroup_matrix_block, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19, !20}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fa"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fb"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fc"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"ha"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hc"}
!17 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!18 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fs"}
!19 = !{i32 7, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hs"}
!20 = !{i32 8, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!21 = !{!"air.compile.denorms_disable"}
!22 = !{!"air.compile.fast_math_enable"}
!23 = !{!"air.compile.framebuffer_fetch_enable"}
!24 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!25 = !{i32 2, i32 8, i32 0}
!26 = !{!"Metal", i32 4, i32 0, i32 0}
!27 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_simdgroup_matrix_block.metal"}
!28 = !{!29, !29, i64 0}
!29 = !{!"float", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C++ TBAA"}
!32 = !{!33}
!33 = distinct !{!33, !34, !"air-alias-scope-arg(6)"}
!34 = distinct !{!34, !"air-alias-scopes(kernel_simdgroup_matrix_block)"}
!35 = !{!36, !37, !38, !39, !40, !41, !42}
!36 = distinct !{!36, !34, !"air-alias-scope-arg(0)"}
!37 = distinct !{!37, !34, !"air-alias-scope-arg(1)"}
!38 = distinct !{!38, !34, !"air-alias-scope-arg(2)"}
!39 = distinct !{!39, !34, !"air-alias-scope-arg(3)"}
!40 = distinct !{!40, !34, !"air-alias-scope-arg(4)"}
!41 = distinct !{!41, !34, !"air-alias-scope-arg(5)"}
!42 = distinct !{!42, !34, !"air-alias-scope-arg(7)"}
!43 = !{!44, !44, i64 0}
!44 = !{!"int", !30, i64 0}
!45 = !{!41}
!46 = !{!36, !37, !38, !39, !40, !33, !42}
!47 = !{!48, !48, i64 0}
!48 = !{!"half", !30, i64 0}
!49 = !{!42}
!50 = !{!36, !37, !38, !39, !40, !41, !33}
