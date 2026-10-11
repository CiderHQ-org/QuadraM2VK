; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'cvf.bc'
source_filename = "validation/fixtures/public/kernel_convert_fma_fract.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_convert_fma_fract(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3) local_unnamed_addr #0 {
  %5 = load float, ptr addrspace(1) %1, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %6 = insertelement <4 x float> undef, float %5, i64 0
  %7 = getelementptr inbounds float, ptr addrspace(1) %1, i64 1
  %8 = load float, ptr addrspace(1) %7, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %9 = insertelement <4 x float> %6, float %8, i64 1
  %10 = getelementptr inbounds float, ptr addrspace(1) %1, i64 2
  %11 = load float, ptr addrspace(1) %10, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %12 = insertelement <4 x float> %9, float %11, i64 2
  %13 = getelementptr inbounds float, ptr addrspace(1) %1, i64 3
  %14 = load float, ptr addrspace(1) %13, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %15 = insertelement <4 x float> %12, float %14, i64 3
  %16 = tail call <4 x i8> @air.convert.u.v4i8.f.v4f32(<4 x float> %15) #2
  %17 = load i16, ptr addrspace(1) %2, align 2, !tbaa !34, !alias.scope !36, !noalias !37
  %18 = insertelement <3 x i16> undef, i16 %17, i64 0
  %19 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 1
  %20 = load i16, ptr addrspace(1) %19, align 2, !tbaa !34, !alias.scope !36, !noalias !37
  %21 = insertelement <3 x i16> %18, i16 %20, i64 1
  %22 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 2
  %23 = load i16, ptr addrspace(1) %22, align 2, !tbaa !34, !alias.scope !36, !noalias !37
  %24 = insertelement <3 x i16> %21, i16 %23, i64 2
  %25 = tail call <3 x i32> @air.convert.s.v3i32.u.v3i16(<3 x i16> %24) #2
  %26 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 3
  %27 = load i16, ptr addrspace(1) %26, align 2, !tbaa !34, !alias.scope !36, !noalias !37
  %28 = icmp ne i16 %27, 0
  %29 = insertelement <2 x i1> undef, i1 %28, i64 0
  %30 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 4
  %31 = load i16, ptr addrspace(1) %30, align 2, !tbaa !34, !alias.scope !36, !noalias !37
  %32 = icmp ne i16 %31, 0
  %33 = insertelement <2 x i1> %29, i1 %32, i64 1
  %34 = tail call fast <2 x half> @air.convert.f.v2f16.u.v2i1(<2 x i1> %33) #2
  %35 = load half, ptr addrspace(1) %3, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %36 = insertelement <3 x half> undef, half %35, i64 0
  %37 = getelementptr inbounds half, ptr addrspace(1) %3, i64 1
  %38 = load half, ptr addrspace(1) %37, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %39 = insertelement <3 x half> %36, half %38, i64 1
  %40 = getelementptr inbounds half, ptr addrspace(1) %3, i64 2
  %41 = load half, ptr addrspace(1) %40, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %42 = insertelement <3 x half> %39, half %41, i64 2
  %43 = getelementptr inbounds half, ptr addrspace(1) %3, i64 3
  %44 = load half, ptr addrspace(1) %43, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %45 = insertelement <3 x half> undef, half %44, i64 0
  %46 = getelementptr inbounds half, ptr addrspace(1) %3, i64 4
  %47 = load half, ptr addrspace(1) %46, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %48 = insertelement <3 x half> %45, half %47, i64 1
  %49 = getelementptr inbounds half, ptr addrspace(1) %3, i64 5
  %50 = load half, ptr addrspace(1) %49, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %51 = insertelement <3 x half> %48, half %50, i64 2
  %52 = getelementptr inbounds half, ptr addrspace(1) %3, i64 6
  %53 = load half, ptr addrspace(1) %52, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %54 = insertelement <3 x half> undef, half %53, i64 0
  %55 = getelementptr inbounds half, ptr addrspace(1) %3, i64 7
  %56 = load half, ptr addrspace(1) %55, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %57 = insertelement <3 x half> %54, half %56, i64 1
  %58 = getelementptr inbounds half, ptr addrspace(1) %3, i64 8
  %59 = load half, ptr addrspace(1) %58, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %60 = insertelement <3 x half> %57, half %59, i64 2
  %61 = tail call fast <3 x half> @air.fma.v3f16(<3 x half> %42, <3 x half> %51, <3 x half> %60) #2
  %62 = getelementptr inbounds half, ptr addrspace(1) %3, i64 9
  %63 = load half, ptr addrspace(1) %62, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %64 = insertelement <3 x half> undef, half %63, i64 0
  %65 = getelementptr inbounds half, ptr addrspace(1) %3, i64 10
  %66 = load half, ptr addrspace(1) %65, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %67 = insertelement <3 x half> %64, half %66, i64 1
  %68 = getelementptr inbounds half, ptr addrspace(1) %3, i64 11
  %69 = load half, ptr addrspace(1) %68, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %70 = insertelement <3 x half> %67, half %69, i64 2
  %71 = tail call fast <3 x half> @air.fract.v3f16(<3 x half> %70) #2
  %72 = extractelement <4 x i8> %16, i64 0
  %73 = zext i8 %72 to i32
  store i32 %73, ptr addrspace(1) %0, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %74 = extractelement <4 x i8> %16, i64 1
  %75 = zext i8 %74 to i32
  %76 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %75, ptr addrspace(1) %76, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %77 = extractelement <4 x i8> %16, i64 2
  %78 = zext i8 %77 to i32
  %79 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  store i32 %78, ptr addrspace(1) %79, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %80 = extractelement <4 x i8> %16, i64 3
  %81 = zext i8 %80 to i32
  %82 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  store i32 %81, ptr addrspace(1) %82, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %83 = extractelement <3 x i32> %25, i64 0
  %84 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  store i32 %83, ptr addrspace(1) %84, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %85 = extractelement <3 x i32> %25, i64 1
  %86 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  store i32 %85, ptr addrspace(1) %86, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %87 = extractelement <3 x i32> %25, i64 2
  %88 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  store i32 %87, ptr addrspace(1) %88, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %89 = bitcast <2 x half> %34 to <2 x i16>
  %90 = extractelement <2 x i16> %89, i64 0
  %91 = zext i16 %90 to i32
  %92 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  store i32 %91, ptr addrspace(1) %92, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %93 = extractelement <2 x i16> %89, i64 1
  %94 = zext i16 %93 to i32
  %95 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 8
  store i32 %94, ptr addrspace(1) %95, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %96 = bitcast <3 x half> %61 to <3 x i16>
  %97 = extractelement <3 x i16> %96, i64 0
  %98 = zext i16 %97 to i32
  %99 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 9
  store i32 %98, ptr addrspace(1) %99, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %100 = extractelement <3 x i16> %96, i64 1
  %101 = zext i16 %100 to i32
  %102 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 10
  store i32 %101, ptr addrspace(1) %102, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %103 = extractelement <3 x i16> %96, i64 2
  %104 = zext i16 %103 to i32
  %105 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 11
  store i32 %104, ptr addrspace(1) %105, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %106 = bitcast <3 x half> %71 to <3 x i16>
  %107 = extractelement <3 x i16> %106, i64 0
  %108 = zext i16 %107 to i32
  %109 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 12
  store i32 %108, ptr addrspace(1) %109, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %110 = extractelement <3 x i16> %106, i64 1
  %111 = zext i16 %110 to i32
  %112 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 13
  store i32 %111, ptr addrspace(1) %112, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %113 = extractelement <3 x i16> %106, i64 2
  %114 = zext i16 %113 to i32
  %115 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 14
  store i32 %114, ptr addrspace(1) %115, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i8> @air.convert.u.v4i8.f.v4f32(<4 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i32> @air.convert.s.v3i32.u.v3i16(<3 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x half> @air.convert.f.v2f16.u.v2i1(<2 x i1>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x half> @air.fma.v3f16(<3 x half>, <3 x half>, <3 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x half> @air.fract.v3f16(<3 x half>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="48" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!16, !17, !18}
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
!9 = !{ptr @kernel_convert_fma_fract, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort", !"air.arg_name", !"sin_"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hin"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_convert_fma_fract.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"float", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(1)"}
!29 = distinct !{!29, !"air-alias-scopes(kernel_convert_fma_fract)"}
!30 = !{!31, !32, !33}
!31 = distinct !{!31, !29, !"air-alias-scope-arg(0)"}
!32 = distinct !{!32, !29, !"air-alias-scope-arg(2)"}
!33 = distinct !{!33, !29, !"air-alias-scope-arg(3)"}
!34 = !{!35, !35, i64 0}
!35 = !{!"short", !25, i64 0}
!36 = !{!32}
!37 = !{!31, !28, !33}
!38 = !{!39, !39, i64 0}
!39 = !{!"half", !25, i64 0}
!40 = !{!33}
!41 = !{!31, !28, !32}
!42 = !{!43, !43, i64 0}
!43 = !{!"int", !25, i64 0}
!44 = !{!31}
!45 = !{!28, !32, !33}
