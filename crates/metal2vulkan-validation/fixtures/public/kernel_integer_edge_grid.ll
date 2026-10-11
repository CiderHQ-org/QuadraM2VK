; ModuleID = '/tmp/ie.bc'
source_filename = "validation/fixtures/public/kernel_integer_edge_grid.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_integer_edge_grid(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = urem i32 %2, 12
  %5 = zext i32 %4 to i64
  %6 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %5
  %7 = load i32, ptr addrspace(1) %6, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %8 = add i32 %2, 1
  %9 = urem i32 %8, 12
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %10
  %12 = load i32, ptr addrspace(1) %11, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %13 = add i32 %2, 2
  %14 = urem i32 %13, 12
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %15
  %17 = load i32, ptr addrspace(1) %16, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %18 = mul i32 %2, 17
  %19 = tail call i32 @air.clz.i32(i32 %7, i1 false) #2
  %20 = zext i32 %18 to i64
  %21 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %20
  store i32 %19, ptr addrspace(1) %21, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %22 = tail call i32 @air.ctz.i32(i32 %7, i1 false) #2
  %23 = add i32 %18, 1
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %24
  store i32 %22, ptr addrspace(1) %25, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %26 = tail call i32 @air.popcount.i32(i32 %7) #2
  %27 = add i32 %18, 2
  %28 = zext i32 %27 to i64
  %29 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %28
  store i32 %26, ptr addrspace(1) %29, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %30 = tail call i32 @air.abs.s.i32(i32 %7) #2
  %31 = add i32 %18, 3
  %32 = zext i32 %31 to i64
  %33 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %32
  store i32 %30, ptr addrspace(1) %33, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %34 = tail call i32 @air.reverse_bits.i32(i32 %7) #2
  %35 = add i32 %18, 4
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %36
  store i32 %34, ptr addrspace(1) %37, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %38 = and i32 %12, 31
  %39 = tail call i32 @air.rotate.i32(i32 %7, i32 %38) #2
  %40 = add i32 %18, 5
  %41 = zext i32 %40 to i64
  %42 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %41
  store i32 %39, ptr addrspace(1) %42, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %43 = tail call i32 @air.mul_hi.u.i32(i32 %7, i32 %12) #2
  %44 = add i32 %18, 6
  %45 = zext i32 %44 to i64
  %46 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %45
  store i32 %43, ptr addrspace(1) %46, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %47 = tail call i32 @air.mad_sat.s.i32(i32 %7, i32 %12, i32 %17) #2
  %48 = add i32 %18, 7
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %49
  store i32 %47, ptr addrspace(1) %50, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %51 = and i32 %12, 15
  %52 = and i32 %17, 15
  %53 = add nuw nsw i32 %52, 1
  %54 = tail call i32 @air.extract_bits.u.i32(i32 %7, i32 %51, i32 %53) #2
  %55 = add i32 %18, 8
  %56 = zext i32 %55 to i64
  %57 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %56
  store i32 %54, ptr addrspace(1) %57, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %58 = tail call i32 @air.extract_bits.s.i32(i32 %7, i32 %51, i32 %53) #2
  %59 = add i32 %18, 9
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %60
  store i32 %58, ptr addrspace(1) %61, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %62 = and i32 %7, 7
  %63 = add nuw nsw i32 %62, 1
  %64 = tail call i32 @air.insert_bits.u.i32(i32 %7, i32 %12, i32 %52, i32 %63) #2
  %65 = add i32 %18, 10
  %66 = zext i32 %65 to i64
  %67 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %66
  store i32 %64, ptr addrspace(1) %67, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %68 = tail call i32 @air.extract_bits.u.i32(i32 %7, i32 16, i32 16) #2
  %69 = add i32 %18, 11
  %70 = zext i32 %69 to i64
  %71 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %70
  store i32 %68, ptr addrspace(1) %71, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %72 = tail call i32 @air.insert_bits.u.i32(i32 %7, i32 %12, i32 24, i32 8) #2
  %73 = add i32 %18, 12
  %74 = zext i32 %73 to i64
  %75 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %74
  store i32 %72, ptr addrspace(1) %75, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %76 = trunc i32 %7 to i16
  %77 = trunc i32 %12 to i16
  %78 = tail call i16 @air.rhadd.u.i16(i16 %76, i16 %77) #2
  %79 = zext i16 %78 to i32
  %80 = add i32 %18, 13
  %81 = zext i32 %80 to i64
  %82 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %81
  store i32 %79, ptr addrspace(1) %82, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %83 = trunc i32 %7 to i8
  %84 = trunc i32 %12 to i8
  %85 = tail call i8 @air.add_sat.u.i8(i8 %83, i8 %84) #2
  %86 = zext i8 %85 to i32
  %87 = add i32 %18, 14
  %88 = zext i32 %87 to i64
  %89 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %88
  store i32 %86, ptr addrspace(1) %89, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %90 = tail call i8 @air.sub_sat.u.i8(i8 %83, i8 %84) #2
  %91 = zext i8 %90 to i32
  %92 = add i32 %18, 15
  %93 = zext i32 %92 to i64
  %94 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %93
  store i32 %91, ptr addrspace(1) %94, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %95 = tail call i32 @air.min.s.i32(i32 %12, i32 %17) #2
  %96 = tail call i32 @air.max.s.i32(i32 %12, i32 %17) #2
  %97 = tail call i32 @air.clamp.s.i32(i32 %7, i32 %95, i32 %96) #2
  %98 = add i32 %18, 16
  %99 = zext i32 %98 to i64
  %100 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %99
  store i32 %97, ptr addrspace(1) %100, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.clz.i32(i32, i1) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.ctz.i32(i32, i1) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.popcount.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.abs.s.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.reverse_bits.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.rotate.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.mul_hi.u.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.mad_sat.s.i32(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.extract_bits.u.i32(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.extract_bits.s.i32(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.insert_bits.u.i32(i32, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.rhadd.u.i16(i16, i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i8 @air.add_sat.u.i8(i8, i8) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i8 @air.sub_sat.u.i8(i8, i8) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.clamp.s.i32(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.min.s.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.max.s.i32(i32, i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_integer_edge_grid, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 3, i32 1, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_integer_edge_grid.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_integer_edge_grid)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
