; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'vc2.bc'
source_filename = "validation/fixtures/public/kernel_narrow_vector_converts.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_narrow_vector_converts(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %4, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %5, i32 noundef %6) local_unnamed_addr #0 {
  %8 = zext i32 %6 to i64
  %9 = getelementptr inbounds <4 x float>, ptr addrspace(1) %0, i64 %8
  %10 = load <4 x float>, ptr addrspace(1) %9, align 16, !tbaa !26, !alias.scope !29, !noalias !32
  %11 = tail call <4 x i64> @air.convert.u.v4i64.f.v4f32(<4 x float> %10) #2
  %12 = getelementptr inbounds <2 x i32>, ptr addrspace(1) %1, i64 %8
  %13 = load <2 x i32>, ptr addrspace(1) %12, align 8, !tbaa !26, !alias.scope !38, !noalias !39
  %14 = tail call <2 x i8> @air.convert.u.v2i8.u.v2i32(<2 x i32> %13) #2
  %15 = getelementptr inbounds <2 x i64>, ptr addrspace(1) %2, i64 %8
  %16 = load <2 x i64>, ptr addrspace(1) %15, align 16, !tbaa !26, !alias.scope !40, !noalias !41
  %17 = tail call <2 x i32> @air.convert.u.v2i32.u.v2i64(<2 x i64> %16) #2
  %18 = getelementptr inbounds <2 x i32>, ptr addrspace(1) %3, i64 %8
  %19 = load <2 x i32>, ptr addrspace(1) %18, align 8, !tbaa !26, !alias.scope !42, !noalias !43
  %20 = tail call fast <2 x half> @air.convert.f.v2f16.s.v2i32(<2 x i32> %19) #2
  %21 = getelementptr inbounds <4 x i16>, ptr addrspace(1) %4, i64 %8
  %22 = load <4 x i16>, ptr addrspace(1) %21, align 8, !alias.scope !44, !noalias !45
  %23 = shufflevector <4 x i16> %22, <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %24 = tail call <3 x i8> @air.convert.u.v3i8.s.v3i16(<3 x i16> %23) #2
  %25 = tail call <3 x i16> @air.convert.s.v3i16.u.v3i8(<3 x i8> %24) #2
  %26 = tail call <2 x i16> @air.convert.s.v2i16.u.v2i32(<2 x i32> %13) #2
  %27 = shufflevector <4 x i16> %22, <4 x i16> poison, <2 x i32> <i32 0, i32 1>
  %28 = tail call <2 x i8> @air.convert.s.v2i8.s.v2i16(<2 x i16> %27) #2
  %29 = tail call <2 x i16> @air.convert.s.v2i16.s.v2i8(<2 x i8> %28) #2
  %30 = extractelement <2 x i32> %19, i64 0
  %31 = tail call fast half @air.convert.f.f16.s.i32(i32 %30) #2
  %32 = extractelement <2 x i32> %13, i64 0
  %33 = tail call fast half @air.convert.f.f16.u.i32(i32 %32) #2
  %34 = mul i32 %6, 13
  %35 = extractelement <4 x i64> %11, i64 0
  %36 = trunc i64 %35 to i32
  %37 = zext i32 %34 to i64
  %38 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %37
  store i32 %36, ptr addrspace(1) %38, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %39 = lshr i64 %35, 32
  %40 = trunc i64 %39 to i32
  %41 = add i32 %34, 1
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %42
  store i32 %40, ptr addrspace(1) %43, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %44 = extractelement <4 x i64> %11, i64 3
  %45 = trunc i64 %44 to i32
  %46 = add i32 %34, 2
  %47 = zext i32 %46 to i64
  %48 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %47
  store i32 %45, ptr addrspace(1) %48, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %49 = lshr i64 %44, 32
  %50 = trunc i64 %49 to i32
  %51 = add i32 %34, 3
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %52
  store i32 %50, ptr addrspace(1) %53, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %54 = extractelement <2 x i8> %14, i64 0
  %55 = zext i8 %54 to i32
  %56 = extractelement <2 x i8> %14, i64 1
  %57 = zext i8 %56 to i32
  %58 = shl nuw nsw i32 %57, 8
  %59 = or i32 %58, %55
  %60 = add i32 %34, 4
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %61
  store i32 %59, ptr addrspace(1) %62, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %63 = extractelement <2 x i32> %17, i64 0
  %64 = add i32 %34, 5
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %65
  store i32 %63, ptr addrspace(1) %66, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %67 = extractelement <2 x i32> %17, i64 1
  %68 = add i32 %34, 6
  %69 = zext i32 %68 to i64
  %70 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %69
  store i32 %67, ptr addrspace(1) %70, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %71 = bitcast <2 x half> %20 to <2 x i16>
  %72 = extractelement <2 x i16> %71, i64 0
  %73 = zext i16 %72 to i32
  %74 = extractelement <2 x i16> %71, i64 1
  %75 = zext i16 %74 to i32
  %76 = shl nuw i32 %75, 16
  %77 = or i32 %76, %73
  %78 = add i32 %34, 7
  %79 = zext i32 %78 to i64
  %80 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %79
  store i32 %77, ptr addrspace(1) %80, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %81 = extractelement <3 x i8> %24, i64 0
  %82 = zext i8 %81 to i32
  %83 = extractelement <3 x i8> %24, i64 1
  %84 = zext i8 %83 to i32
  %85 = shl nuw nsw i32 %84, 8
  %86 = or i32 %85, %82
  %87 = extractelement <3 x i8> %24, i64 2
  %88 = zext i8 %87 to i32
  %89 = shl nuw nsw i32 %88, 16
  %90 = or i32 %86, %89
  %91 = add i32 %34, 8
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %92
  store i32 %90, ptr addrspace(1) %93, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %94 = extractelement <3 x i16> %25, i64 0
  %95 = zext i16 %94 to i32
  %96 = extractelement <3 x i16> %25, i64 1
  %97 = zext i16 %96 to i32
  %98 = shl nuw i32 %97, 16
  %99 = or i32 %98, %95
  %100 = add i32 %34, 9
  %101 = zext i32 %100 to i64
  %102 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %101
  store i32 %99, ptr addrspace(1) %102, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %103 = extractelement <2 x i16> %26, i64 0
  %104 = zext i16 %103 to i32
  %105 = extractelement <2 x i16> %26, i64 1
  %106 = zext i16 %105 to i32
  %107 = shl nuw i32 %106, 16
  %108 = or i32 %107, %104
  %109 = add i32 %34, 10
  %110 = zext i32 %109 to i64
  %111 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %110
  store i32 %108, ptr addrspace(1) %111, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %112 = extractelement <2 x i16> %29, i64 0
  %113 = zext i16 %112 to i32
  %114 = extractelement <2 x i16> %29, i64 1
  %115 = zext i16 %114 to i32
  %116 = shl nuw i32 %115, 16
  %117 = or i32 %116, %113
  %118 = add i32 %34, 11
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %119
  store i32 %117, ptr addrspace(1) %120, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  %121 = bitcast half %31 to i16
  %122 = zext i16 %121 to i32
  %123 = bitcast half %33 to i16
  %124 = zext i16 %123 to i32
  %125 = shl nuw i32 %124, 16
  %126 = or i32 %125, %122
  %127 = add i32 %34, 12
  %128 = zext i32 %127 to i64
  %129 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %128
  store i32 %126, ptr addrspace(1) %129, align 4, !tbaa !46, !alias.scope !48, !noalias !49
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i64> @air.convert.u.v4i64.f.v4f32(<4 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i8> @air.convert.u.v2i8.u.v2i32(<2 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i32> @air.convert.u.v2i32.u.v2i64(<2 x i64>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x half> @air.convert.f.v2f16.s.v2i32(<2 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i8> @air.convert.u.v3i8.s.v3i16(<3 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i16> @air.convert.s.v3i16.u.v3i8(<3 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i16> @air.convert.s.v2i16.u.v2i32(<2 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i8> @air.convert.s.v2i8.s.v2i16(<2 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i16> @air.convert.s.v2i16.s.v2i8(<2 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.s.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i32(i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_narrow_vector_converts, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"fin"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"uint2", !"air.arg_name", !"uin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"ulong2", !"air.arg_name", !"lin"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"int2", !"air.arg_name", !"iin"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"short4", !"air.arg_name", !"hin"}
!17 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!18 = !{i32 6, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 4, i32 0, i32 0}
!25 = !{!"/private/tmp/mkfix/vc2.metal"}
!26 = !{!27, !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C++ TBAA"}
!29 = !{!30}
!30 = distinct !{!30, !31, !"air-alias-scope-arg(0)"}
!31 = distinct !{!31, !"air-alias-scopes(kernel_narrow_vector_converts)"}
!32 = !{!33, !34, !35, !36, !37}
!33 = distinct !{!33, !31, !"air-alias-scope-arg(1)"}
!34 = distinct !{!34, !31, !"air-alias-scope-arg(2)"}
!35 = distinct !{!35, !31, !"air-alias-scope-arg(3)"}
!36 = distinct !{!36, !31, !"air-alias-scope-arg(4)"}
!37 = distinct !{!37, !31, !"air-alias-scope-arg(5)"}
!38 = !{!33}
!39 = !{!30, !34, !35, !36, !37}
!40 = !{!34}
!41 = !{!30, !33, !35, !36, !37}
!42 = !{!35}
!43 = !{!30, !33, !34, !36, !37}
!44 = !{!36}
!45 = !{!30, !33, !34, !35, !37}
!46 = !{!47, !47, i64 0}
!47 = !{!"int", !27, i64 0}
!48 = !{!37}
!49 = !{!30, !33, !34, !35, !36}
