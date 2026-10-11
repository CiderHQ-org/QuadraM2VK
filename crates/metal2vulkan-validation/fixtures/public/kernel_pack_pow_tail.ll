; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'pk.bc'
source_filename = "validation/fixtures/public/kernel_pack_pow_tail.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_pack_pow_tail(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3) local_unnamed_addr #0 {
  %5 = load half, ptr addrspace(1) %1, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %6 = insertelement <4 x half> undef, half %5, i64 0
  %7 = getelementptr inbounds half, ptr addrspace(1) %1, i64 1
  %8 = load half, ptr addrspace(1) %7, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %9 = insertelement <4 x half> %6, half %8, i64 1
  %10 = getelementptr inbounds half, ptr addrspace(1) %1, i64 2
  %11 = load half, ptr addrspace(1) %10, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %12 = insertelement <4 x half> %9, half %11, i64 2
  %13 = getelementptr inbounds half, ptr addrspace(1) %1, i64 3
  %14 = load half, ptr addrspace(1) %13, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %15 = insertelement <4 x half> %12, half %14, i64 3
  %16 = getelementptr inbounds half, ptr addrspace(1) %1, i64 4
  %17 = load half, ptr addrspace(1) %16, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %18 = insertelement <4 x half> undef, half %17, i64 0
  %19 = getelementptr inbounds half, ptr addrspace(1) %1, i64 5
  %20 = load half, ptr addrspace(1) %19, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %21 = insertelement <4 x half> %18, half %20, i64 1
  %22 = getelementptr inbounds half, ptr addrspace(1) %1, i64 6
  %23 = load half, ptr addrspace(1) %22, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %24 = insertelement <4 x half> %21, half %23, i64 2
  %25 = getelementptr inbounds half, ptr addrspace(1) %1, i64 7
  %26 = load half, ptr addrspace(1) %25, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %27 = insertelement <4 x half> %24, half %26, i64 3
  %28 = tail call fast <4 x half> @air.pow.v4f16(<4 x half> %15, <4 x half> %27) #2
  %29 = getelementptr inbounds half, ptr addrspace(1) %1, i64 8
  %30 = load half, ptr addrspace(1) %29, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %31 = insertelement <2 x half> undef, half %30, i64 0
  %32 = getelementptr inbounds half, ptr addrspace(1) %1, i64 9
  %33 = load half, ptr addrspace(1) %32, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %34 = insertelement <2 x half> %31, half %33, i64 1
  %35 = getelementptr inbounds half, ptr addrspace(1) %1, i64 10
  %36 = load half, ptr addrspace(1) %35, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %37 = insertelement <2 x half> undef, half %36, i64 0
  %38 = getelementptr inbounds half, ptr addrspace(1) %1, i64 11
  %39 = load half, ptr addrspace(1) %38, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %40 = insertelement <2 x half> %37, half %39, i64 1
  %41 = tail call fast <2 x half> @air.powr.v2f16(<2 x half> %34, <2 x half> %40) #2
  %42 = load float, ptr addrspace(1) %2, align 4, !tbaa !34, !alias.scope !36, !noalias !37
  %43 = insertelement <2 x float> undef, float %42, i64 0
  %44 = getelementptr inbounds float, ptr addrspace(1) %2, i64 1
  %45 = load float, ptr addrspace(1) %44, align 4, !tbaa !34, !alias.scope !36, !noalias !37
  %46 = insertelement <2 x float> %43, float %45, i64 1
  %47 = getelementptr inbounds float, ptr addrspace(1) %2, i64 2
  %48 = load float, ptr addrspace(1) %47, align 4, !tbaa !34, !alias.scope !36, !noalias !37
  %49 = insertelement <2 x float> undef, float %48, i64 0
  %50 = getelementptr inbounds float, ptr addrspace(1) %2, i64 3
  %51 = load float, ptr addrspace(1) %50, align 4, !tbaa !34, !alias.scope !36, !noalias !37
  %52 = insertelement <2 x float> %49, float %51, i64 1
  %53 = tail call fast <2 x float> @air.fast_pow.v2f32(<2 x float> %46, <2 x float> %52) #2
  %54 = getelementptr inbounds half, ptr addrspace(1) %1, i64 12
  %55 = load half, ptr addrspace(1) %54, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %56 = insertelement <2 x half> undef, half %55, i64 0
  %57 = getelementptr inbounds half, ptr addrspace(1) %1, i64 13
  %58 = load half, ptr addrspace(1) %57, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %59 = insertelement <2 x half> %56, half %58, i64 1
  %60 = tail call i32 @air.pack.unorm2x16.v2f16(<2 x half> %59) #2
  %61 = getelementptr inbounds half, ptr addrspace(1) %1, i64 14
  %62 = load half, ptr addrspace(1) %61, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %63 = insertelement <3 x half> undef, half %62, i64 0
  %64 = getelementptr inbounds half, ptr addrspace(1) %1, i64 15
  %65 = load half, ptr addrspace(1) %64, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %66 = insertelement <3 x half> %63, half %65, i64 1
  %67 = getelementptr inbounds half, ptr addrspace(1) %1, i64 16
  %68 = load half, ptr addrspace(1) %67, align 2, !tbaa !23, !alias.scope !27, !noalias !30
  %69 = insertelement <3 x half> %66, half %68, i64 2
  %70 = tail call i16 @air.pack.unorm.rgb565.v3f16(<3 x half> %69) #2
  %71 = load i32, ptr addrspace(1) %3, align 4, !tbaa !38, !alias.scope !40, !noalias !41
  %72 = tail call fast <2 x float> @air.unpack.unorm2x16.v2f32(i32 %71) #2
  %73 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 1
  %74 = load i32, ptr addrspace(1) %73, align 4, !tbaa !38, !alias.scope !40, !noalias !41
  %75 = tail call fast <2 x half> @air.unpack.unorm2x16.v2f16(i32 %74) #2
  %76 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 2
  %77 = load i32, ptr addrspace(1) %76, align 4, !tbaa !38, !alias.scope !40, !noalias !41
  %78 = tail call fast <4 x half> @air.unpack.unorm.rgb10a2.v4f16(i32 %77) #2
  %79 = bitcast <4 x half> %28 to <4 x i16>
  %80 = extractelement <4 x i16> %79, i64 0
  %81 = zext i16 %80 to i32
  store i32 %81, ptr addrspace(1) %0, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %82 = extractelement <4 x i16> %79, i64 1
  %83 = zext i16 %82 to i32
  %84 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %83, ptr addrspace(1) %84, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %85 = extractelement <4 x i16> %79, i64 2
  %86 = zext i16 %85 to i32
  %87 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  store i32 %86, ptr addrspace(1) %87, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %88 = extractelement <4 x i16> %79, i64 3
  %89 = zext i16 %88 to i32
  %90 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  store i32 %89, ptr addrspace(1) %90, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %91 = bitcast <2 x half> %41 to <2 x i16>
  %92 = extractelement <2 x i16> %91, i64 0
  %93 = zext i16 %92 to i32
  %94 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  store i32 %93, ptr addrspace(1) %94, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %95 = extractelement <2 x i16> %91, i64 1
  %96 = zext i16 %95 to i32
  %97 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  store i32 %96, ptr addrspace(1) %97, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %98 = bitcast <2 x float> %53 to <2 x i32>
  %99 = extractelement <2 x i32> %98, i64 0
  %100 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  store i32 %99, ptr addrspace(1) %100, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %101 = extractelement <2 x i32> %98, i64 1
  %102 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  store i32 %101, ptr addrspace(1) %102, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %103 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 8
  store i32 %60, ptr addrspace(1) %103, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %104 = zext i16 %70 to i32
  %105 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 9
  store i32 %104, ptr addrspace(1) %105, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %106 = bitcast <2 x float> %72 to <2 x i32>
  %107 = extractelement <2 x i32> %106, i64 0
  %108 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 10
  store i32 %107, ptr addrspace(1) %108, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %109 = extractelement <2 x i32> %106, i64 1
  %110 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 11
  store i32 %109, ptr addrspace(1) %110, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %111 = bitcast <2 x half> %75 to <2 x i16>
  %112 = extractelement <2 x i16> %111, i64 0
  %113 = zext i16 %112 to i32
  %114 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 12
  store i32 %113, ptr addrspace(1) %114, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %115 = extractelement <2 x i16> %111, i64 1
  %116 = zext i16 %115 to i32
  %117 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 13
  store i32 %116, ptr addrspace(1) %117, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %118 = bitcast <4 x half> %78 to <4 x i16>
  %119 = extractelement <4 x i16> %118, i64 0
  %120 = zext i16 %119 to i32
  %121 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 14
  store i32 %120, ptr addrspace(1) %121, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %122 = extractelement <4 x i16> %118, i64 1
  %123 = zext i16 %122 to i32
  %124 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 15
  store i32 %123, ptr addrspace(1) %124, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %125 = extractelement <4 x i16> %118, i64 2
  %126 = zext i16 %125 to i32
  %127 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 16
  store i32 %126, ptr addrspace(1) %127, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  %128 = extractelement <4 x i16> %118, i64 3
  %129 = zext i16 %128 to i32
  %130 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 17
  store i32 %129, ptr addrspace(1) %130, align 4, !tbaa !38, !alias.scope !42, !noalias !43
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.pow.v4f16(<4 x half>, <4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x half> @air.powr.v2f16(<2 x half>, <2 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_pow.v2f32(<2 x float>, <2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.pack.unorm2x16.v2f16(<2 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.pack.unorm.rgb565.v3f16(<3 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.unpack.unorm2x16.v2f32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x half> @air.unpack.unorm2x16.v2f16(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.unpack.unorm.rgb10a2.v4f16(i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_pack_pow_tail, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fin"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"uin"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_pack_pow_tail.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"half", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(1)"}
!29 = distinct !{!29, !"air-alias-scopes(kernel_pack_pow_tail)"}
!30 = !{!31, !32, !33}
!31 = distinct !{!31, !29, !"air-alias-scope-arg(0)"}
!32 = distinct !{!32, !29, !"air-alias-scope-arg(2)"}
!33 = distinct !{!33, !29, !"air-alias-scope-arg(3)"}
!34 = !{!35, !35, i64 0}
!35 = !{!"float", !25, i64 0}
!36 = !{!32}
!37 = !{!31, !28, !33}
!38 = !{!39, !39, i64 0}
!39 = !{!"int", !25, i64 0}
!40 = !{!33}
!41 = !{!31, !28, !32}
!42 = !{!31}
!43 = !{!28, !32, !33}
