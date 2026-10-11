; ModuleID = '/tmp/ms/f.bc'
source_filename = "validation/fixtures/public/kernel_multisample_reads.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nounwind willreturn
define void @kernel_multisample_reads(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, i32 noundef %6) local_unnamed_addr #0 {
  %8 = and i32 %6, 1
  %9 = insertelement <2 x i32> undef, i32 %8, i64 0
  %10 = lshr i32 %6, 1
  %11 = insertelement <2 x i32> %9, i32 %10, i64 1
  %12 = trunc i32 %8 to i16
  %13 = insertelement <2 x i16> undef, i16 %12, i64 0
  %14 = trunc i32 %10 to i16
  %15 = insertelement <2 x i16> %13, i16 %14, i64 1
  %16 = tail call ptr addrspace(2) @air.get_read_sampler() #3
  %17 = tail call { <4 x i16>, i8 } @air.read_texture_2d_ms.u.v4i16(ptr addrspace(1) readonly captures(none) %1, ptr addrspace(2) %16, <2 x i32> %11, i32 %8, i32 1) #4
  %18 = extractvalue { <4 x i16>, i8 } %17, 0
  %19 = tail call { <4 x i32>, i8 } @air.read_texture_2d_ms.u.v4i32(ptr addrspace(1) readonly captures(none) %2, ptr addrspace(2) %16, <2 x i32> %11, i32 %8, i32 1) #4
  %20 = extractvalue { <4 x i32>, i8 } %19, 0
  %21 = tail call { <4 x i32>, i8 } @air.read_texture_2d_ms.s.v4i32(ptr addrspace(1) readonly captures(none) %3, ptr addrspace(2) %16, <2 x i32> %11, i32 %8, i32 1) #4
  %22 = extractvalue { <4 x i32>, i8 } %21, 0
  %23 = tail call { <4 x half>, i8 } @air.read_texture_2d_ms.i16.v4f16(ptr addrspace(1) readonly captures(none) %4, ptr addrspace(2) %16, <2 x i16> %15, i16 %12, i32 1) #4
  %24 = extractvalue { <4 x half>, i8 } %23, 0
  %25 = tail call { <4 x i16>, i8 } @air.read_texture_2d_ms.i16.u.v4i16(ptr addrspace(1) readonly captures(none) %1, ptr addrspace(2) %16, <2 x i16> %15, i16 %12, i32 1) #4
  %26 = extractvalue { <4 x i16>, i8 } %25, 0
  %27 = tail call { float, i8 } @air.read_depth_2d_ms.f32(ptr addrspace(1) readonly captures(none) %5, ptr addrspace(2) %16, i32 1, <2 x i32> %11, i32 %8, i32 1) #4
  %28 = extractvalue { float, i8 } %27, 0
  %29 = extractelement <4 x i16> %18, i64 0
  %30 = zext i16 %29 to i32
  %31 = mul i32 %6, 6
  %32 = zext i32 %31 to i64
  %33 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %32
  store i32 %30, ptr addrspace(1) %33, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %34 = extractelement <4 x i32> %20, i64 1
  %35 = or i32 %31, 1
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %36
  store i32 %34, ptr addrspace(1) %37, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %38 = extractelement <4 x i32> %22, i64 2
  %39 = add i32 %31, 2
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %40
  store i32 %38, ptr addrspace(1) %41, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %42 = bitcast <4 x half> %24 to <4 x i16>
  %43 = extractelement <4 x i16> %42, i64 3
  %44 = zext i16 %43 to i32
  %45 = add i32 %31, 3
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %46
  store i32 %44, ptr addrspace(1) %47, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %48 = extractelement <4 x i16> %26, i64 1
  %49 = zext i16 %48 to i32
  %50 = add i32 %31, 4
  %51 = zext i32 %50 to i64
  %52 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %51
  store i32 %49, ptr addrspace(1) %52, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %53 = add i32 %31, 5
  %54 = zext i32 %53 to i64
  %55 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %54
  %56 = bitcast ptr addrspace(1) %55 to ptr addrspace(1)
  store float %28, ptr addrspace(1) %56, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: read)
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i16>, i8 } @air.read_texture_2d_ms.u.v4i16(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i32>, i8 } @air.read_texture_2d_ms.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i32>, i8 } @air.read_texture_2d_ms.s.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x half>, i8 } @air.read_texture_2d_ms.i16.v4f16(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i16>, i16, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x i16>, i8 } @air.read_texture_2d_ms.i16.u.v4i16(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i16>, i16, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { float, i8 } @air.read_depth_2d_ms.f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), i32, <2 x i32>, i32, i32) local_unnamed_addr #2

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #4 = { nounwind willreturn memory(argmem: read) }

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
!9 = !{ptr @kernel_multisample_reads, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<ushort, read>", !"air.arg_name", !"msu16"}
!14 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<uint, read>", !"air.arg_name", !"msu32"}
!15 = !{i32 3, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<int, read>", !"air.arg_name", !"mss32"}
!16 = !{i32 4, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<half, read>", !"air.arg_name", !"msh"}
!17 = !{i32 5, !"air.texture", !"air.location_index", i32 4, i32 1, !"air.read", !"air.arg_type_name", !"depth2d_ms<float, read>", !"air.arg_name", !"msd"}
!18 = !{i32 6, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 3, i32 2, i32 0}
!25 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_multisample_reads.metal"}
!26 = !{!27, !27, i64 0}
!27 = !{!"int", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"air-alias-scope-arg(0)"}
!32 = distinct !{!32, !"air-alias-scopes(kernel_multisample_reads)"}
!33 = !{!34}
!34 = distinct !{!34, !32, !"air-alias-scope-textures"}
