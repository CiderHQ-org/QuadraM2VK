; Hand edit (DESIGN D149): the `llvm.lifetime.*` calls and declarations take the two
; arguments LLVM 21's `llvm-dis` writes (`i64 <alloca size in bytes>, ptr`), which LLVM 22+
; auto-upgrades by dropping the size, so every gate's `llvm-as` accepts the text.
; ModuleID = '/tmp/b2/f.bc'
source_filename = "validation/fixtures/public/kernel_simd_quad_and_null_texture.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_simd_quad_and_null_texture(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) readonly captures(none) %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = alloca half, align 2
  %6 = zext i32 %3 to i64
  %7 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %6
  %8 = bitcast ptr addrspace(1) %7 to ptr addrspace(1)
  %9 = load float, ptr addrspace(1) %8, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %10 = fptrunc float %9 to half
  %11 = bitcast ptr %5 to ptr
  call void @llvm.lifetime.start.p0(i64 2, ptr %5)
  %12 = call fast half @air.sincos.f16(half %10, ptr nonnull writeonly captures(none) %5) #7
  %13 = tail call fast half @air.simd_max.f16(half %10) #8
  %14 = tail call fast half @air.simd_min.f16(half %10) #8
  %15 = trunc i32 %3 to i16
  %16 = add i16 %15, -10
  %17 = tail call i16 @air.simd_min.s.i16(i16 %16) #8
  %18 = tail call i16 @air.simd_broadcast.s.i16(i16 %16, i16 0) #8
  %19 = and i32 %3, 7
  %20 = shl nuw nsw i32 1, %19
  %21 = trunc i32 %20 to i8
  %22 = tail call i8 @air.simd_or.u.i8(i8 %21) #8
  %23 = icmp eq i32 %3, 7
  %24 = tail call i1 @air.simd_any(i1 %23) #8
  %25 = tail call i32 @air.simd_prefix_exclusive_sum.s.i32(i32 %3) #8
  %26 = tail call fast half @air.convert.f.f16.u.i32(i32 %3) #9
  %27 = insertelement <3 x half> undef, half %26, i64 0
  %28 = fmul fast half %26, 0xH4000
  %29 = insertelement <3 x half> %27, half %28, i64 1
  %30 = fmul fast half %26, 0xH4400
  %31 = insertelement <3 x half> %29, half %30, i64 2
  %32 = tail call fast <3 x half> @air.quad_sum.v3f16(<3 x half> %31) #8
  %33 = insertelement <2 x half> undef, half %26, i64 0
  %34 = fmul fast half %26, 0xH4800
  %35 = insertelement <2 x half> %33, half %34, i64 1
  %36 = tail call fast <2 x half> @air.quad_sum.v2f16(<2 x half> %35) #8
  %37 = tail call fast half @air.quad_sum.f16(half %26) #8
  %38 = fadd fast half %10, 0xHC3C0
  %39 = fsub fast half 0xH43C0, %10
  %40 = insertelement <4 x half> <half poison, half poison, half 0xH4000, half 0xHC000>, half %38, i64 0
  %41 = insertelement <4 x half> %40, half %39, i64 1
  %42 = tail call fast <4 x half> @air.sign.v4f16(<4 x half> %41) #9
  %43 = tail call ptr addrspace(1) @air.get_null_texture_3d() #10
  %44 = shl i32 %3, 4
  %45 = bitcast half %12 to i16
  %46 = zext i16 %45 to i32
  %47 = zext i32 %44 to i64
  %48 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %47
  store i32 %46, ptr addrspace(1) %48, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %49 = bitcast ptr %5 to ptr
  %50 = load i16, ptr %49, align 2, !tbaa !35
  %51 = zext i16 %50 to i32
  %52 = or i32 %44, 1
  %53 = zext i32 %52 to i64
  %54 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %53
  store i32 %51, ptr addrspace(1) %54, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %55 = bitcast half %13 to i16
  %56 = zext i16 %55 to i32
  %57 = or i32 %44, 2
  %58 = zext i32 %57 to i64
  %59 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %58
  store i32 %56, ptr addrspace(1) %59, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %60 = bitcast half %14 to i16
  %61 = zext i16 %60 to i32
  %62 = or i32 %44, 3
  %63 = zext i32 %62 to i64
  %64 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %63
  store i32 %61, ptr addrspace(1) %64, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %65 = sext i16 %17 to i32
  %66 = or i32 %44, 4
  %67 = zext i32 %66 to i64
  %68 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %67
  store i32 %65, ptr addrspace(1) %68, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %69 = sext i16 %18 to i32
  %70 = or i32 %44, 5
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %71
  store i32 %69, ptr addrspace(1) %72, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %73 = zext i8 %22 to i32
  %74 = or i32 %44, 6
  %75 = zext i32 %74 to i64
  %76 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %75
  store i32 %73, ptr addrspace(1) %76, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %77 = zext i1 %24 to i32
  %78 = or i32 %44, 7
  %79 = zext i32 %78 to i64
  %80 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %79
  store i32 %77, ptr addrspace(1) %80, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %81 = or i32 %44, 8
  %82 = zext i32 %81 to i64
  %83 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %82
  store i32 %25, ptr addrspace(1) %83, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %84 = bitcast <3 x half> %32 to <3 x i16>
  %85 = extractelement <3 x i16> %84, i64 2
  %86 = zext i16 %85 to i32
  %87 = or i32 %44, 9
  %88 = zext i32 %87 to i64
  %89 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %88
  store i32 %86, ptr addrspace(1) %89, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %90 = bitcast <2 x half> %36 to <2 x i16>
  %91 = extractelement <2 x i16> %90, i64 1
  %92 = zext i16 %91 to i32
  %93 = or i32 %44, 10
  %94 = zext i32 %93 to i64
  %95 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %94
  store i32 %92, ptr addrspace(1) %95, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %96 = bitcast half %37 to i16
  %97 = zext i16 %96 to i32
  %98 = or i32 %44, 11
  %99 = zext i32 %98 to i64
  %100 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %99
  store i32 %97, ptr addrspace(1) %100, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %101 = bitcast <4 x half> %42 to <4 x i16>
  %102 = extractelement <4 x i16> %101, i64 0
  %103 = zext i16 %102 to i32
  %104 = or i32 %44, 12
  %105 = zext i32 %104 to i64
  %106 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %105
  store i32 %103, ptr addrspace(1) %106, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %107 = extractelement <4 x i16> %101, i64 1
  %108 = zext i16 %107 to i32
  %109 = or i32 %44, 13
  %110 = zext i32 %109 to i64
  %111 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %110
  store i32 %108, ptr addrspace(1) %111, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %112 = tail call i1 @air.is_null_texture_3d(ptr addrspace(1) readonly captures(none) %43) #11
  %113 = zext i1 %112 to i32
  %114 = or i32 %44, 14
  %115 = zext i32 %114 to i64
  %116 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %115
  store i32 %113, ptr addrspace(1) %116, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  %117 = tail call i1 @air.is_null_texture_3d(ptr addrspace(1) readonly captures(none) %2) #11, !alias.scope !37, !noalias !38
  %118 = zext i1 %117 to i32
  %119 = or i32 %44, 15
  %120 = zext i32 %119 to i64
  %121 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %120
  store i32 %118, ptr addrspace(1) %121, align 4, !tbaa !23, !alias.scope !33, !noalias !34
  call void @llvm.lifetime.end.p0(i64 2, ptr %5)
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare half @air.sincos.f16(half, ptr writeonly captures(none)) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare i1 @air.simd_any(i1) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.sign.v4f16(<4 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare half @air.simd_max.f16(half) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn
declare half @air.simd_min.f16(half) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn
declare i16 @air.simd_min.s.i16(i16) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn
declare i16 @air.simd_broadcast.s.i16(i16, i16) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn
declare i8 @air.simd_or.u.i8(i8) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.simd_prefix_exclusive_sum.s.i32(i32) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x half> @air.quad_sum.v3f16(<3 x half>) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x half> @air.quad_sum.v2f16(<2 x half>) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn
declare half @air.quad_sum.f16(half) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: read)
declare ptr addrspace(1) @air.get_null_texture_3d() local_unnamed_addr #4

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i1 @air.is_null_texture_3d(ptr addrspace(1) readonly captures(none)) local_unnamed_addr #5

declare void @llvm.lifetime.start.i64(i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #6

declare void @llvm.lifetime.end.i64(i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #6

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { convergent mustprogress nounwind willreturn }
attributes #4 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #5 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nounwind willreturn memory(argmem: readwrite) }
attributes #8 = { convergent nounwind willreturn }
attributes #9 = { nounwind willreturn memory(none) }
attributes #10 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #11 = { nounwind willreturn memory(argmem: read) }

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
!9 = !{ptr @kernel_simd_quad_and_null_texture, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!14 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture3d<float, sample>", !"air.arg_name", !"bound"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 3, i32 2, i32 0}
!22 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_simd_quad_and_null_texture.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(1)"}
!29 = distinct !{!29, !"air-alias-scopes(kernel_simd_quad_and_null_texture)"}
!30 = !{!31, !32}
!31 = distinct !{!31, !29, !"air-alias-scope-arg(0)"}
!32 = distinct !{!32, !29, !"air-alias-scope-textures"}
!33 = !{!31}
!34 = !{!28, !32}
!35 = !{!36, !36, i64 0}
!36 = !{!"half", !25, i64 0}
!37 = !{!32}
!38 = !{!31, !28}
