; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. Fifteen `air.quad_*`
; type variants that the corpus reaches once each and that no authored case covered.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_quad_variant_leftovers.bc'
source_filename = "validation/fixtures/public/kernel_quad_variant_leftovers.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_quad_variant_leftovers(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = zext i32 %5 to i64
  %8 = getelementptr inbounds <4 x float>, ptr addrspace(1) %0, i64 %7
  %9 = load <4 x float>, ptr addrspace(1) %8, align 16, !tbaa !25, !alias.scope !28, !noalias !31
  %10 = getelementptr inbounds <4 x half>, ptr addrspace(1) %1, i64 %7
  %11 = load <4 x half>, ptr addrspace(1) %10, align 8, !tbaa !25, !alias.scope !36, !noalias !37
  %12 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %7
  %13 = load <4 x i32>, ptr addrspace(1) %12, align 16, !tbaa !25, !alias.scope !38, !noalias !39
  %14 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 %7
  %15 = load i32, ptr addrspace(1) %14, align 4, !tbaa !40, !alias.scope !42, !noalias !43
  %16 = mul i32 %5, 23
  %17 = zext i32 %16 to i64
  %18 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %17
  %19 = tail call i32 @air.quad_sum.s.i32(i32 %15) #2
  store i32 %19, ptr addrspace(1) %18, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %20 = shufflevector <4 x float> %9, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %21 = freeze <2 x float> %20
  %22 = tail call fast <2 x float> @air.quad_shuffle_xor.v2f32(<2 x float> %21, i16 1) #2
  %23 = bitcast <2 x float> %22 to <2 x i32>
  %24 = extractelement <2 x i32> %23, i64 0
  %25 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 1
  store i32 %24, ptr addrspace(1) %25, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %26 = extractelement <2 x i32> %23, i64 1
  %27 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 2
  store i32 %26, ptr addrspace(1) %27, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %28 = freeze <4 x float> %9
  %29 = extractelement <4 x float> %28, i64 0
  %30 = tail call fast float @air.quad_shuffle_up.f32(float %29, i16 1) #2
  %31 = icmp eq i32 %5, 0
  %32 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 3
  br i1 %31, label %35, label %33

33:                                               ; preds = %6
  %34 = bitcast ptr addrspace(1) %32 to ptr addrspace(1)
  store float %30, ptr addrspace(1) %34, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  br label %36

35:                                               ; preds = %6
  store i32 -1412584499, ptr addrspace(1) %32, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  br label %36

36:                                               ; preds = %35, %33
  %37 = trunc i32 %5 to i16
  %38 = sub i16 3, %37
  %39 = tail call fast float @air.quad_shuffle_rotate_down.f32(float %29, i16 1) #2
  %40 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 4
  %41 = bitcast ptr addrspace(1) %40 to ptr addrspace(1)
  store float %39, ptr addrspace(1) %41, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %42 = freeze <4 x half> %11
  %43 = tail call fast <4 x half> @air.quad_shuffle.v4f16(<4 x half> %42, i16 %38) #2
  %44 = bitcast <4 x half> %43 to <4 x i16>
  %45 = extractelement <4 x i16> %44, i64 0
  %46 = zext i16 %45 to i32
  %47 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 5
  store i32 %46, ptr addrspace(1) %47, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %48 = extractelement <4 x i16> %44, i64 1
  %49 = zext i16 %48 to i32
  %50 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 6
  store i32 %49, ptr addrspace(1) %50, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %51 = extractelement <4 x i16> %44, i64 2
  %52 = zext i16 %51 to i32
  %53 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 7
  store i32 %52, ptr addrspace(1) %53, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %54 = extractelement <4 x i16> %44, i64 3
  %55 = zext i16 %54 to i32
  %56 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 8
  store i32 %55, ptr addrspace(1) %56, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %57 = shufflevector <4 x float> %28, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %58 = tail call fast <2 x float> @air.quad_shuffle.v2f32(<2 x float> %57, i16 %38) #2
  %59 = bitcast <2 x float> %58 to <2 x i32>
  %60 = extractelement <2 x i32> %59, i64 0
  %61 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 9
  store i32 %60, ptr addrspace(1) %61, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %62 = extractelement <2 x i32> %59, i64 1
  %63 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 10
  store i32 %62, ptr addrspace(1) %63, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %64 = freeze <4 x i32> %13
  %65 = extractelement <4 x i32> %64, i64 0
  %66 = tail call i32 @air.quad_shuffle.u.i32(i32 %65, i16 %38) #2
  %67 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 11
  store i32 %66, ptr addrspace(1) %67, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %68 = extractelement <4 x i32> %64, i64 1
  %69 = tail call i32 @air.quad_min.u.i32(i32 %68) #2
  %70 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 12
  store i32 %69, ptr addrspace(1) %70, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %71 = tail call i32 @air.quad_max.u.i32(i32 %68) #2
  %72 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 13
  store i32 %71, ptr addrspace(1) %72, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %73 = tail call fast <2 x float> @air.quad_broadcast.v2f32(<2 x float> %21, i16 2) #2
  %74 = bitcast <2 x float> %73 to <2 x i32>
  %75 = extractelement <2 x i32> %74, i64 0
  %76 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 14
  store i32 %75, ptr addrspace(1) %76, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %77 = extractelement <2 x i32> %74, i64 1
  %78 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 15
  store i32 %77, ptr addrspace(1) %78, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %79 = shufflevector <4 x half> %42, <4 x half> poison, <2 x i32> <i32 0, i32 1>
  %80 = tail call fast <2 x half> @air.quad_broadcast.v2f16(<2 x half> %79, i16 2) #2
  %81 = bitcast <2 x half> %80 to <2 x i16>
  %82 = extractelement <2 x i16> %81, i64 0
  %83 = zext i16 %82 to i32
  %84 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 16
  store i32 %83, ptr addrspace(1) %84, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %85 = extractelement <2 x i16> %81, i64 1
  %86 = zext i16 %85 to i32
  %87 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 17
  store i32 %86, ptr addrspace(1) %87, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %88 = extractelement <4 x i32> %64, i64 2
  %89 = trunc i32 %88 to i16
  %90 = insertelement <2 x i16> poison, i16 %89, i64 0
  %91 = extractelement <4 x i32> %64, i64 3
  %92 = trunc i32 %91 to i16
  %93 = insertelement <2 x i16> %90, i16 %92, i64 1
  %94 = tail call <2 x i16> @air.quad_broadcast.u.v2i16(<2 x i16> %93, i16 2) #2
  %95 = extractelement <2 x i16> %94, i64 0
  %96 = zext i16 %95 to i32
  %97 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 18
  store i32 %96, ptr addrspace(1) %97, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %98 = extractelement <2 x i16> %94, i64 1
  %99 = zext i16 %98 to i32
  %100 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 19
  store i32 %99, ptr addrspace(1) %100, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %101 = tail call i32 @air.quad_broadcast.u.i32(i32 %88, i16 2) #2
  %102 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 20
  store i32 %101, ptr addrspace(1) %102, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %103 = tail call i16 @air.quad_broadcast.u.i16(i16 %92, i16 2) #2
  %104 = zext i16 %103 to i32
  %105 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 21
  store i32 %104, ptr addrspace(1) %105, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  %106 = extractelement <4 x half> %42, i64 2
  %107 = tail call fast half @air.quad_broadcast.f16(half %106, i16 2) #2
  %108 = bitcast half %107 to i16
  %109 = zext i16 %108 to i32
  %110 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 22
  store i32 %109, ptr addrspace(1) %110, align 4, !tbaa !40, !alias.scope !44, !noalias !45
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.quad_sum.s.i32(i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x float> @air.quad_shuffle_xor.v2f32(<2 x float>, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.quad_shuffle_up.f32(float, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.quad_shuffle_rotate_down.f32(float, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x half> @air.quad_shuffle.v4f16(<4 x half>, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x float> @air.quad_shuffle.v2f32(<2 x float>, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.quad_shuffle.u.i32(i32, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.quad_min.u.i32(i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.quad_max.u.i32(i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x float> @air.quad_broadcast.v2f32(<2 x float>, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x half> @air.quad_broadcast.v2f16(<2 x half>, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x i16> @air.quad_broadcast.u.v2i16(<2 x i16>, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.quad_broadcast.u.i32(i32, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i16 @air.quad_broadcast.u.i16(i16, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare half @air.quad_broadcast.f16(half, i16) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_quad_variant_leftovers, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"fin"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"hin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"uin"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"iin"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!17 = !{i32 5, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
!24 = !{!"/private/tmp/quad/k.metal"}
!25 = !{!26, !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(0)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_quad_variant_leftovers)"}
!31 = !{!32, !33, !34, !35}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(1)"}
!33 = distinct !{!33, !30, !"air-alias-scope-arg(2)"}
!34 = distinct !{!34, !30, !"air-alias-scope-arg(3)"}
!35 = distinct !{!35, !30, !"air-alias-scope-arg(4)"}
!36 = !{!32}
!37 = !{!29, !33, !34, !35}
!38 = !{!33}
!39 = !{!29, !32, !34, !35}
!40 = !{!41, !41, i64 0}
!41 = !{!"int", !26, i64 0}
!42 = !{!34}
!43 = !{!29, !32, !33, !35}
!44 = !{!35}
!45 = !{!29, !32, !33, !34}
