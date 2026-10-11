; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'u.bc'
source_filename = "validation/fixtures/public/kernel_unpack_packed_pixels.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%"struct.metal::_pixel_type" = type { i32 }
%"struct.metal::_pixel_type.0" = type { i32 }

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_unpack_packed_pixels(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3, i32 noundef %4) local_unnamed_addr #0 {
  %6 = load i32, ptr addrspace(1) %1, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %7 = tail call fast <4 x float> @air.unpack.unorm.rgb10a2.v4f32(i32 %6) #2
  %8 = getelementptr inbounds %"struct.metal::_pixel_type", ptr addrspace(1) %2, i64 0, i32 0
  %9 = load i32, ptr addrspace(1) %8, align 4, !tbaa !35, !alias.scope !37, !noalias !38
  %10 = tail call fast <3 x float> @air.unpack.unorm.rg11b10f.v3f32(i32 %9) #2
  %11 = getelementptr inbounds %"struct.metal::_pixel_type.0", ptr addrspace(1) %3, i64 0, i32 0
  %12 = load i32, ptr addrspace(1) %11, align 4, !tbaa !39, !alias.scope !41, !noalias !42
  %13 = tail call fast <3 x float> @air.unpack.unorm.rgb9e5.v3f32(i32 %12) #2
  %14 = extractelement <4 x float> %7, i64 0
  store float %14, ptr addrspace(1) %0, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %15 = extractelement <4 x float> %7, i64 1
  %16 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  store float %15, ptr addrspace(1) %16, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %17 = extractelement <4 x float> %7, i64 2
  %18 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  store float %17, ptr addrspace(1) %18, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %19 = extractelement <4 x float> %7, i64 3
  %20 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  store float %19, ptr addrspace(1) %20, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %21 = extractelement <3 x float> %10, i64 0
  %22 = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  store float %21, ptr addrspace(1) %22, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %23 = extractelement <3 x float> %10, i64 1
  %24 = getelementptr inbounds float, ptr addrspace(1) %0, i64 5
  store float %23, ptr addrspace(1) %24, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %25 = extractelement <3 x float> %10, i64 2
  %26 = getelementptr inbounds float, ptr addrspace(1) %0, i64 6
  store float %25, ptr addrspace(1) %26, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %27 = extractelement <3 x float> %13, i64 0
  %28 = getelementptr inbounds float, ptr addrspace(1) %0, i64 7
  store float %27, ptr addrspace(1) %28, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %29 = extractelement <3 x float> %13, i64 1
  %30 = getelementptr inbounds float, ptr addrspace(1) %0, i64 8
  store float %29, ptr addrspace(1) %30, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %31 = extractelement <3 x float> %13, i64 2
  %32 = getelementptr inbounds float, ptr addrspace(1) %0, i64 9
  store float %31, ptr addrspace(1) %32, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x float> @air.unpack.unorm.rgb10a2.v4f32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.unpack.unorm.rg11b10f.v3f32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.unpack.unorm.rgb9e5.v3f32(i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}
!air.source_file_name = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_unpack_packed_pixels, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"rg11b10f<float3>", !"air.arg_name", !"fpix"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"rgb9e5<float3>", !"air.arg_name", !"epix"}
!16 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid", !"air.arg_unused"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 4, i32 0, i32 0}
!23 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_unpack_packed_pixels.metal"}
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(1)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_unpack_packed_pixels)"}
!31 = !{!32, !33, !34}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(0)"}
!33 = distinct !{!33, !30, !"air-alias-scope-arg(2)"}
!34 = distinct !{!34, !30, !"air-alias-scope-arg(3)"}
!35 = !{!36, !25, i64 0}
!36 = !{!"_ZTSN5metal11_pixel_typeIDv3_fNS_13_rg11b10f_tagEvEE", !25, i64 0}
!37 = !{!33}
!38 = !{!32, !29, !34}
!39 = !{!40, !25, i64 0}
!40 = !{!"_ZTSN5metal11_pixel_typeIDv3_fNS_11_rgb9e5_tagEvEE", !25, i64 0}
!41 = !{!34}
!42 = !{!32, !29, !33}
!43 = !{!44, !44, i64 0}
!44 = !{!"float", !26, i64 0}
!45 = !{!32}
!46 = !{!29, !33, !34}
