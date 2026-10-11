; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. The plain calls
; already give the `air.exp`/`air.log`/`air.log2` spellings at half width; no `precise::` is
; needed, because at half width the fast and precise variants are the same bits.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_half4_exp_and_logs.bc'
source_filename = "validation/fixtures/public/kernel_half4_exp_and_logs.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_half4_exp_and_logs(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = load <4 x half>, ptr addrspace(1) %0, align 8, !tbaa !21, !alias.scope !24, !noalias !27
  %4 = tail call fast <4 x half> @air.exp.v4f16(<4 x half> %3) #2
  %5 = getelementptr inbounds <4 x half>, ptr addrspace(1) %0, i64 1
  %6 = load <4 x half>, ptr addrspace(1) %5, align 8, !tbaa !21, !alias.scope !24, !noalias !27
  %7 = tail call fast <4 x half> @air.exp.v4f16(<4 x half> %6) #2
  %8 = getelementptr inbounds <4 x half>, ptr addrspace(1) %0, i64 2
  %9 = load <4 x half>, ptr addrspace(1) %8, align 8, !tbaa !21, !alias.scope !24, !noalias !27
  %10 = tail call fast <4 x half> @air.log.v4f16(<4 x half> %9) #2
  %11 = getelementptr inbounds <4 x half>, ptr addrspace(1) %0, i64 3
  %12 = load <4 x half>, ptr addrspace(1) %11, align 8, !tbaa !21, !alias.scope !24, !noalias !27
  %13 = tail call fast <4 x half> @air.log.v4f16(<4 x half> %12) #2
  %14 = getelementptr inbounds <4 x half>, ptr addrspace(1) %0, i64 4
  %15 = load <4 x half>, ptr addrspace(1) %14, align 8, !tbaa !21, !alias.scope !24, !noalias !27
  %16 = tail call fast <4 x half> @air.log2.v4f16(<4 x half> %15) #2
  %17 = getelementptr inbounds <4 x half>, ptr addrspace(1) %0, i64 5
  %18 = load <4 x half>, ptr addrspace(1) %17, align 8, !tbaa !21, !alias.scope !24, !noalias !27
  %19 = tail call fast <4 x half> @air.log2.v4f16(<4 x half> %18) #2
  %20 = bitcast <4 x half> %4 to <4 x i16>
  %21 = extractelement <4 x i16> %20, i64 0
  %22 = zext i16 %21 to i32
  store i32 %22, ptr addrspace(1) %1, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %23 = extractelement <4 x i16> %20, i64 1
  %24 = zext i16 %23 to i32
  %25 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 1
  store i32 %24, ptr addrspace(1) %25, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %26 = extractelement <4 x i16> %20, i64 2
  %27 = zext i16 %26 to i32
  %28 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 2
  store i32 %27, ptr addrspace(1) %28, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %29 = extractelement <4 x i16> %20, i64 3
  %30 = zext i16 %29 to i32
  %31 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 3
  store i32 %30, ptr addrspace(1) %31, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %32 = bitcast <4 x half> %7 to <4 x i16>
  %33 = extractelement <4 x i16> %32, i64 0
  %34 = zext i16 %33 to i32
  %35 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 4
  store i32 %34, ptr addrspace(1) %35, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %36 = extractelement <4 x i16> %32, i64 1
  %37 = zext i16 %36 to i32
  %38 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 5
  store i32 %37, ptr addrspace(1) %38, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %39 = extractelement <4 x i16> %32, i64 2
  %40 = zext i16 %39 to i32
  %41 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 6
  store i32 %40, ptr addrspace(1) %41, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %42 = extractelement <4 x i16> %32, i64 3
  %43 = zext i16 %42 to i32
  %44 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 7
  store i32 %43, ptr addrspace(1) %44, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %45 = bitcast <4 x half> %10 to <4 x i16>
  %46 = extractelement <4 x i16> %45, i64 0
  %47 = zext i16 %46 to i32
  %48 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 8
  store i32 %47, ptr addrspace(1) %48, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %49 = extractelement <4 x i16> %45, i64 1
  %50 = zext i16 %49 to i32
  %51 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 9
  store i32 %50, ptr addrspace(1) %51, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %52 = extractelement <4 x i16> %45, i64 2
  %53 = zext i16 %52 to i32
  %54 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 10
  store i32 %53, ptr addrspace(1) %54, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %55 = extractelement <4 x i16> %45, i64 3
  %56 = zext i16 %55 to i32
  %57 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 11
  store i32 %56, ptr addrspace(1) %57, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %58 = bitcast <4 x half> %13 to <4 x i16>
  %59 = extractelement <4 x i16> %58, i64 0
  %60 = zext i16 %59 to i32
  %61 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 12
  store i32 %60, ptr addrspace(1) %61, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %62 = extractelement <4 x i16> %58, i64 1
  %63 = zext i16 %62 to i32
  %64 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 13
  store i32 %63, ptr addrspace(1) %64, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %65 = extractelement <4 x i16> %58, i64 2
  %66 = zext i16 %65 to i32
  %67 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 14
  store i32 %66, ptr addrspace(1) %67, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %68 = extractelement <4 x i16> %58, i64 3
  %69 = zext i16 %68 to i32
  %70 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 15
  store i32 %69, ptr addrspace(1) %70, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %71 = bitcast <4 x half> %16 to <4 x i16>
  %72 = extractelement <4 x i16> %71, i64 0
  %73 = zext i16 %72 to i32
  %74 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 16
  store i32 %73, ptr addrspace(1) %74, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %75 = extractelement <4 x i16> %71, i64 1
  %76 = zext i16 %75 to i32
  %77 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 17
  store i32 %76, ptr addrspace(1) %77, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %78 = extractelement <4 x i16> %71, i64 2
  %79 = zext i16 %78 to i32
  %80 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 18
  store i32 %79, ptr addrspace(1) %80, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %81 = extractelement <4 x i16> %71, i64 3
  %82 = zext i16 %81 to i32
  %83 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 19
  store i32 %82, ptr addrspace(1) %83, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %84 = bitcast <4 x half> %19 to <4 x i16>
  %85 = extractelement <4 x i16> %84, i64 0
  %86 = zext i16 %85 to i32
  %87 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 20
  store i32 %86, ptr addrspace(1) %87, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %88 = extractelement <4 x i16> %84, i64 1
  %89 = zext i16 %88 to i32
  %90 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 21
  store i32 %89, ptr addrspace(1) %90, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %91 = extractelement <4 x i16> %84, i64 2
  %92 = zext i16 %91 to i32
  %93 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 22
  store i32 %92, ptr addrspace(1) %93, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  %94 = extractelement <4 x i16> %84, i64 3
  %95 = zext i16 %94 to i32
  %96 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 23
  store i32 %95, ptr addrspace(1) %96, align 4, !tbaa !29, !alias.scope !27, !noalias !24
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.exp.v4f16(<4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.log.v4f16(<4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.log2.v4f16(<4 x half>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_half4_exp_and_logs, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"v"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/private/tmp/hf/k.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C++ TBAA"}
!24 = !{!25}
!25 = distinct !{!25, !26, !"air-alias-scope-arg(0)"}
!26 = distinct !{!26, !"air-alias-scopes(kernel_half4_exp_and_logs)"}
!27 = !{!28}
!28 = distinct !{!28, !26, !"air-alias-scope-arg(1)"}
!29 = !{!30, !30, i64 0}
!30 = !{!"int", !22, i64 0}
