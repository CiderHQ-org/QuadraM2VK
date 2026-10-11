; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -c
; -fno-fast-math -x metal -` (stdin) and disassembled with LLVM 21's `llvm-dis` (opaque pointers). No
; call carries a fast-math flag, so the signed-zero and infinite rows ask defined questions (DESIGN
; D208). The float rows are spelled `precise::atan2`, which a fast-math compile needs to reach
; `air.atan2.f32`; at half width the plain call gives `air.atan2.f16`.
; Not derived from a third-party metallib.
; ModuleID = '<stdin>'
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_atan2_on_the_axes(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %4) local_unnamed_addr #0 {
  %6 = load half, ptr addrspace(1) %0, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %7 = load half, ptr addrspace(1) %1, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %8 = tail call half @air.atan2.f16(half %6, half %7) #2
  %9 = bitcast half %8 to i16
  %10 = zext i16 %9 to i32
  store i32 %10, ptr addrspace(1) %4, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %11 = getelementptr inbounds half, ptr addrspace(1) %0, i64 1
  %12 = load half, ptr addrspace(1) %11, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %13 = getelementptr inbounds half, ptr addrspace(1) %1, i64 1
  %14 = load half, ptr addrspace(1) %13, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %15 = tail call half @air.atan2.f16(half %12, half %14) #2
  %16 = bitcast half %15 to i16
  %17 = zext i16 %16 to i32
  %18 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 1
  store i32 %17, ptr addrspace(1) %18, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %19 = getelementptr inbounds half, ptr addrspace(1) %0, i64 2
  %20 = load half, ptr addrspace(1) %19, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %21 = getelementptr inbounds half, ptr addrspace(1) %1, i64 2
  %22 = load half, ptr addrspace(1) %21, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %23 = tail call half @air.atan2.f16(half %20, half %22) #2
  %24 = bitcast half %23 to i16
  %25 = zext i16 %24 to i32
  %26 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 2
  store i32 %25, ptr addrspace(1) %26, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %27 = getelementptr inbounds half, ptr addrspace(1) %0, i64 3
  %28 = load half, ptr addrspace(1) %27, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %29 = getelementptr inbounds half, ptr addrspace(1) %1, i64 3
  %30 = load half, ptr addrspace(1) %29, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %31 = tail call half @air.atan2.f16(half %28, half %30) #2
  %32 = bitcast half %31 to i16
  %33 = zext i16 %32 to i32
  %34 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 3
  store i32 %33, ptr addrspace(1) %34, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %35 = getelementptr inbounds half, ptr addrspace(1) %0, i64 4
  %36 = load half, ptr addrspace(1) %35, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %37 = getelementptr inbounds half, ptr addrspace(1) %1, i64 4
  %38 = load half, ptr addrspace(1) %37, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %39 = tail call half @air.atan2.f16(half %36, half %38) #2
  %40 = bitcast half %39 to i16
  %41 = zext i16 %40 to i32
  %42 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 4
  store i32 %41, ptr addrspace(1) %42, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %43 = getelementptr inbounds half, ptr addrspace(1) %0, i64 5
  %44 = load half, ptr addrspace(1) %43, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %45 = getelementptr inbounds half, ptr addrspace(1) %1, i64 5
  %46 = load half, ptr addrspace(1) %45, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %47 = tail call half @air.atan2.f16(half %44, half %46) #2
  %48 = bitcast half %47 to i16
  %49 = zext i16 %48 to i32
  %50 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 5
  store i32 %49, ptr addrspace(1) %50, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %51 = getelementptr inbounds half, ptr addrspace(1) %0, i64 6
  %52 = load half, ptr addrspace(1) %51, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %53 = getelementptr inbounds half, ptr addrspace(1) %1, i64 6
  %54 = load half, ptr addrspace(1) %53, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %55 = tail call half @air.atan2.f16(half %52, half %54) #2
  %56 = bitcast half %55 to i16
  %57 = zext i16 %56 to i32
  %58 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 6
  store i32 %57, ptr addrspace(1) %58, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %59 = getelementptr inbounds half, ptr addrspace(1) %0, i64 7
  %60 = load half, ptr addrspace(1) %59, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %61 = getelementptr inbounds half, ptr addrspace(1) %1, i64 7
  %62 = load half, ptr addrspace(1) %61, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %63 = tail call half @air.atan2.f16(half %60, half %62) #2
  %64 = bitcast half %63 to i16
  %65 = zext i16 %64 to i32
  %66 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 7
  store i32 %65, ptr addrspace(1) %66, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %67 = getelementptr inbounds half, ptr addrspace(1) %0, i64 8
  %68 = load half, ptr addrspace(1) %67, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %69 = getelementptr inbounds half, ptr addrspace(1) %1, i64 8
  %70 = load half, ptr addrspace(1) %69, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %71 = tail call half @air.atan2.f16(half %68, half %70) #2
  %72 = bitcast half %71 to i16
  %73 = zext i16 %72 to i32
  %74 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 8
  store i32 %73, ptr addrspace(1) %74, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %75 = getelementptr inbounds half, ptr addrspace(1) %0, i64 9
  %76 = load half, ptr addrspace(1) %75, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %77 = getelementptr inbounds half, ptr addrspace(1) %1, i64 9
  %78 = load half, ptr addrspace(1) %77, align 2, !tbaa !23, !alias.scope !35, !noalias !36
  %79 = tail call half @air.atan2.f16(half %76, half %78) #2
  %80 = bitcast half %79 to i16
  %81 = zext i16 %80 to i32
  %82 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 9
  store i32 %81, ptr addrspace(1) %82, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %83 = load float, ptr addrspace(1) %2, align 4, !tbaa !41, !alias.scope !43, !noalias !44
  %84 = load float, ptr addrspace(1) %3, align 4, !tbaa !41, !alias.scope !45, !noalias !46
  %85 = tail call float @air.atan2.f32(float %83, float %84) #2
  %86 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 10
  %87 = bitcast ptr addrspace(1) %86 to ptr addrspace(1)
  store float %85, ptr addrspace(1) %87, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %88 = getelementptr inbounds float, ptr addrspace(1) %2, i64 1
  %89 = load float, ptr addrspace(1) %88, align 4, !tbaa !41, !alias.scope !43, !noalias !44
  %90 = getelementptr inbounds float, ptr addrspace(1) %3, i64 1
  %91 = load float, ptr addrspace(1) %90, align 4, !tbaa !41, !alias.scope !45, !noalias !46
  %92 = tail call float @air.atan2.f32(float %89, float %91) #2
  %93 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 11
  %94 = bitcast ptr addrspace(1) %93 to ptr addrspace(1)
  store float %92, ptr addrspace(1) %94, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %95 = getelementptr inbounds float, ptr addrspace(1) %2, i64 2
  %96 = load float, ptr addrspace(1) %95, align 4, !tbaa !41, !alias.scope !43, !noalias !44
  %97 = getelementptr inbounds float, ptr addrspace(1) %3, i64 2
  %98 = load float, ptr addrspace(1) %97, align 4, !tbaa !41, !alias.scope !45, !noalias !46
  %99 = tail call float @air.atan2.f32(float %96, float %98) #2
  %100 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 12
  %101 = bitcast ptr addrspace(1) %100 to ptr addrspace(1)
  store float %99, ptr addrspace(1) %101, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %102 = getelementptr inbounds float, ptr addrspace(1) %2, i64 3
  %103 = load float, ptr addrspace(1) %102, align 4, !tbaa !41, !alias.scope !43, !noalias !44
  %104 = getelementptr inbounds float, ptr addrspace(1) %3, i64 3
  %105 = load float, ptr addrspace(1) %104, align 4, !tbaa !41, !alias.scope !45, !noalias !46
  %106 = tail call float @air.atan2.f32(float %103, float %105) #2
  %107 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 13
  %108 = bitcast ptr addrspace(1) %107 to ptr addrspace(1)
  store float %106, ptr addrspace(1) %108, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %109 = getelementptr inbounds float, ptr addrspace(1) %2, i64 4
  %110 = load float, ptr addrspace(1) %109, align 4, !tbaa !41, !alias.scope !43, !noalias !44
  %111 = getelementptr inbounds float, ptr addrspace(1) %3, i64 4
  %112 = load float, ptr addrspace(1) %111, align 4, !tbaa !41, !alias.scope !45, !noalias !46
  %113 = tail call float @air.atan2.f32(float %110, float %112) #2
  %114 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 14
  %115 = bitcast ptr addrspace(1) %114 to ptr addrspace(1)
  store float %113, ptr addrspace(1) %115, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %116 = getelementptr inbounds float, ptr addrspace(1) %2, i64 5
  %117 = load float, ptr addrspace(1) %116, align 4, !tbaa !41, !alias.scope !43, !noalias !44
  %118 = getelementptr inbounds float, ptr addrspace(1) %3, i64 5
  %119 = load float, ptr addrspace(1) %118, align 4, !tbaa !41, !alias.scope !45, !noalias !46
  %120 = tail call float @air.atan2.f32(float %117, float %119) #2
  %121 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 15
  %122 = bitcast ptr addrspace(1) %121 to ptr addrspace(1)
  store float %120, ptr addrspace(1) %122, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %123 = getelementptr inbounds float, ptr addrspace(1) %2, i64 6
  %124 = load float, ptr addrspace(1) %123, align 4, !tbaa !41, !alias.scope !43, !noalias !44
  %125 = getelementptr inbounds float, ptr addrspace(1) %3, i64 6
  %126 = load float, ptr addrspace(1) %125, align 4, !tbaa !41, !alias.scope !45, !noalias !46
  %127 = tail call float @air.atan2.f32(float %124, float %126) #2
  %128 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 16
  %129 = bitcast ptr addrspace(1) %128 to ptr addrspace(1)
  store float %127, ptr addrspace(1) %129, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %130 = getelementptr inbounds float, ptr addrspace(1) %2, i64 7
  %131 = load float, ptr addrspace(1) %130, align 4, !tbaa !41, !alias.scope !43, !noalias !44
  %132 = getelementptr inbounds float, ptr addrspace(1) %3, i64 7
  %133 = load float, ptr addrspace(1) %132, align 4, !tbaa !41, !alias.scope !45, !noalias !46
  %134 = tail call float @air.atan2.f32(float %131, float %133) #2
  %135 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 17
  %136 = bitcast ptr addrspace(1) %135 to ptr addrspace(1)
  store float %134, ptr addrspace(1) %136, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.atan2.f16(half, half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.atan2.f32(float, float) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_atan2_on_the_axes, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hy"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hx"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fy"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fx"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_disable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 4, i32 0, i32 0}
!23 = !{!24, !24, i64 0}
!24 = !{!"half", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(0)"}
!29 = distinct !{!29, !"air-alias-scopes(kernel_atan2_on_the_axes)"}
!30 = !{!31, !32, !33, !34}
!31 = distinct !{!31, !29, !"air-alias-scope-arg(1)"}
!32 = distinct !{!32, !29, !"air-alias-scope-arg(2)"}
!33 = distinct !{!33, !29, !"air-alias-scope-arg(3)"}
!34 = distinct !{!34, !29, !"air-alias-scope-arg(4)"}
!35 = !{!31}
!36 = !{!28, !32, !33, !34}
!37 = !{!38, !38, i64 0}
!38 = !{!"int", !25, i64 0}
!39 = !{!34}
!40 = !{!28, !31, !32, !33}
!41 = !{!42, !42, i64 0}
!42 = !{!"float", !25, i64 0}
!43 = !{!32}
!44 = !{!28, !31, !33, !34}
!45 = !{!33}
!46 = !{!28, !31, !32, !34}
