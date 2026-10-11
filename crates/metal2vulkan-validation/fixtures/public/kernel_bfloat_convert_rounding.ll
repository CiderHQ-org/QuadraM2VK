; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 's2.bc'
source_filename = "validation/fixtures/public/kernel_bfloat_convert_rounding.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_bfloat_convert_rounding(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = zext i32 %5 to i64
  %8 = getelementptr inbounds <4 x i16>, ptr addrspace(1) %3, i64 %7
  %9 = bitcast ptr addrspace(1) %8 to ptr addrspace(1)
  %10 = load <4 x bfloat>, ptr addrspace(1) %9, align 8, !tbaa !25, !alias.scope !28, !noalias !31
  %11 = extractelement <4 x bfloat> %10, i64 0
  %12 = fneg fast bfloat %11
  %13 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %2, i64 %7
  %14 = load <4 x i32>, ptr addrspace(1) %13, align 16, !tbaa !25, !alias.scope !36, !noalias !37
  %15 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %7
  %16 = load i32, ptr addrspace(1) %15, align 4, !tbaa !38, !alias.scope !40, !noalias !41
  %17 = tail call fast bfloat @air.convert.f.bf16.u.i32(i32 %16) #2
  %18 = getelementptr inbounds i64, ptr addrspace(1) %1, i64 %7
  %19 = load i64, ptr addrspace(1) %18, align 8, !tbaa !42, !alias.scope !44, !noalias !45
  %20 = tail call fast bfloat @air.convert.f.bf16.u.i64(i64 %19) #2
  %21 = tail call fast <4 x bfloat> @air.convert.f.v4bf16.s.v4i32(<4 x i32> %14) #2
  %22 = tail call fast <4 x bfloat> @air.convert.f.v4bf16.u.v4i32(<4 x i32> %14) #2
  %23 = trunc i32 %16 to i16
  %24 = tail call fast bfloat @air.convert.f.bf16.u.i16(i16 %23) #2
  %25 = tail call <4 x i32> @air.convert.s.v4i32.f.v4bf16(<4 x bfloat> %10) #2
  %26 = mul i32 %5, 20
  %27 = bitcast bfloat %17 to i16
  %28 = zext i16 %27 to i32
  %29 = zext i32 %26 to i64
  %30 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %29
  store i32 %28, ptr addrspace(1) %30, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %31 = bitcast bfloat %20 to i16
  %32 = zext i16 %31 to i32
  %33 = or i32 %26, 1
  %34 = zext i32 %33 to i64
  %35 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %34
  store i32 %32, ptr addrspace(1) %35, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %36 = bitcast <4 x bfloat> %21 to <4 x i16>
  %37 = extractelement <4 x i16> %36, i64 0
  %38 = zext i16 %37 to i32
  %39 = or i32 %26, 2
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %40
  store i32 %38, ptr addrspace(1) %41, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %42 = extractelement <4 x i16> %36, i64 1
  %43 = zext i16 %42 to i32
  %44 = or i32 %26, 3
  %45 = zext i32 %44 to i64
  %46 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %45
  store i32 %43, ptr addrspace(1) %46, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %47 = extractelement <4 x i16> %36, i64 2
  %48 = zext i16 %47 to i32
  %49 = add i32 %26, 4
  %50 = zext i32 %49 to i64
  %51 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %50
  store i32 %48, ptr addrspace(1) %51, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %52 = extractelement <4 x i16> %36, i64 3
  %53 = zext i16 %52 to i32
  %54 = add i32 %26, 5
  %55 = zext i32 %54 to i64
  %56 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %55
  store i32 %53, ptr addrspace(1) %56, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %57 = bitcast <4 x bfloat> %22 to <4 x i16>
  %58 = extractelement <4 x i16> %57, i64 0
  %59 = zext i16 %58 to i32
  %60 = add i32 %26, 6
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %61
  store i32 %59, ptr addrspace(1) %62, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %63 = extractelement <4 x i16> %57, i64 1
  %64 = zext i16 %63 to i32
  %65 = add i32 %26, 7
  %66 = zext i32 %65 to i64
  %67 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %66
  store i32 %64, ptr addrspace(1) %67, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %68 = extractelement <4 x i16> %57, i64 2
  %69 = zext i16 %68 to i32
  %70 = add i32 %26, 8
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %71
  store i32 %69, ptr addrspace(1) %72, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %73 = extractelement <4 x i16> %57, i64 3
  %74 = zext i16 %73 to i32
  %75 = add i32 %26, 9
  %76 = zext i32 %75 to i64
  %77 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %76
  store i32 %74, ptr addrspace(1) %77, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %78 = tail call i64 @air.convert.s.i64.f.bf16(bfloat %12) #2
  %79 = trunc i64 %78 to i32
  %80 = add i32 %26, 10
  %81 = zext i32 %80 to i64
  %82 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %81
  store i32 %79, ptr addrspace(1) %82, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %83 = tail call i64 @air.convert.u.i64.f.bf16(bfloat %11) #2
  %84 = trunc i64 %83 to i32
  %85 = add i32 %26, 11
  %86 = zext i32 %85 to i64
  %87 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %86
  store i32 %84, ptr addrspace(1) %87, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %88 = tail call i32 @air.convert.u.i32.f.bf16(bfloat %11) #2
  %89 = add i32 %26, 12
  %90 = zext i32 %89 to i64
  %91 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %90
  store i32 %88, ptr addrspace(1) %91, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %92 = tail call i16 @air.convert.u.i16.f.bf16(bfloat %11) #2
  %93 = zext i16 %92 to i32
  %94 = add i32 %26, 13
  %95 = zext i32 %94 to i64
  %96 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %95
  store i32 %93, ptr addrspace(1) %96, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %97 = tail call i8 @air.convert.u.i8.f.bf16(bfloat %11) #2
  %98 = zext i8 %97 to i32
  %99 = add i32 %26, 14
  %100 = zext i32 %99 to i64
  %101 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %100
  store i32 %98, ptr addrspace(1) %101, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %102 = extractelement <4 x i32> %25, i64 0
  %103 = add i32 %26, 15
  %104 = zext i32 %103 to i64
  %105 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %104
  store i32 %102, ptr addrspace(1) %105, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %106 = extractelement <4 x i32> %25, i64 1
  %107 = add i32 %26, 16
  %108 = zext i32 %107 to i64
  %109 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %108
  store i32 %106, ptr addrspace(1) %109, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %110 = extractelement <4 x i32> %25, i64 2
  %111 = add i32 %26, 17
  %112 = zext i32 %111 to i64
  %113 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %112
  store i32 %110, ptr addrspace(1) %113, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %114 = extractelement <4 x i32> %25, i64 3
  %115 = add i32 %26, 18
  %116 = zext i32 %115 to i64
  %117 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %116
  store i32 %114, ptr addrspace(1) %117, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  %118 = bitcast bfloat %24 to i16
  %119 = zext i16 %118 to i32
  %120 = add i32 %26, 19
  %121 = zext i32 %120 to i64
  %122 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %121
  store i32 %119, ptr addrspace(1) %122, align 4, !tbaa !38, !alias.scope !46, !noalias !47
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare bfloat @air.convert.f.bf16.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare bfloat @air.convert.f.bf16.u.i64(i64) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x bfloat> @air.convert.f.v4bf16.s.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x bfloat> @air.convert.f.v4bf16.u.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare bfloat @air.convert.f.bf16.u.i16(i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.convert.s.v4i32.f.v4bf16(<4 x bfloat>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i64 @air.convert.s.i64.f.bf16(bfloat) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i64 @air.convert.u.i64.f.bf16(bfloat) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.convert.u.i32.f.bf16(bfloat) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.convert.u.i16.f.bf16(bfloat) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i8 @air.convert.u.i8.f.bf16(bfloat) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_bfloat_convert_rounding, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"u32in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"u64in"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"int4", !"air.arg_name", !"i4in"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"bqin"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!17 = !{i32 5, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
!24 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_bfloat_convert_rounding.metal"}
!25 = !{!26, !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(3)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_bfloat_convert_rounding)"}
!31 = !{!32, !33, !34, !35}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(0)"}
!33 = distinct !{!33, !30, !"air-alias-scope-arg(1)"}
!34 = distinct !{!34, !30, !"air-alias-scope-arg(2)"}
!35 = distinct !{!35, !30, !"air-alias-scope-arg(4)"}
!36 = !{!34}
!37 = !{!32, !33, !29, !35}
!38 = !{!39, !39, i64 0}
!39 = !{!"int", !26, i64 0}
!40 = !{!32}
!41 = !{!33, !34, !29, !35}
!42 = !{!43, !43, i64 0}
!43 = !{!"long", !26, i64 0}
!44 = !{!33}
!45 = !{!32, !34, !29, !35}
!46 = !{!35}
!47 = !{!32, !33, !34, !29}
