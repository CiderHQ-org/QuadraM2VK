; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = '/tmp/alias/f.bc'
source_filename = "validation/fixtures/public/kernel_aliased_imageblock_planes.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

; Function Attrs: mustprogress nofree nounwind willreturn
define void @kernel_aliased_imageblock_planes(%"struct.metal::_imageblock_base" %0, <2 x i16> noundef %1) local_unnamed_addr #0 {
  %3 = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %1, i32 0, i16 0) #2
  %4 = bitcast ptr addrspace(4) %3 to ptr addrspace(4)
  %5 = load <4 x half>, ptr addrspace(4) %4, align 8, !tbaa !22
  %6 = getelementptr inbounds i8, ptr addrspace(4) %3, i64 8
  %7 = bitcast ptr addrspace(4) %6 to ptr addrspace(4)
  %8 = load <4 x half>, ptr addrspace(4) %7, align 8, !tbaa !22
  %9 = getelementptr inbounds i8, ptr addrspace(4) %3, i64 16
  %10 = bitcast ptr addrspace(4) %9 to ptr addrspace(4)
  %11 = load half, ptr addrspace(4) %10, align 8, !tbaa !25
  %12 = extractelement <4 x half> %5, i64 0
  %13 = fadd fast half %12, 0xH3C00
  %14 = insertelement <4 x half> undef, half %13, i64 0
  %15 = extractelement <4 x half> %8, i64 1
  %16 = fadd fast half %15, 0xH4000
  %17 = insertelement <4 x half> %14, half %16, i64 1
  %18 = fadd fast half %11, 0xH4200
  %19 = insertelement <4 x half> %17, half %18, i64 2
  %20 = extractelement <4 x half> %5, i64 3
  %21 = fmul fast half %20, 0xH4000
  %22 = insertelement <4 x half> %19, half %21, i64 3
  store <4 x half> %22, ptr addrspace(4) %4, align 8, !tbaa !22
  %23 = extractelement <4 x half> %8, i64 0
  %24 = fmul fast half %23, 0xH4000
  %25 = insertelement <4 x half> undef, half %24, i64 0
  %26 = extractelement <4 x half> %5, i64 1
  %27 = fadd fast half %26, 0xH4400
  %28 = insertelement <4 x half> %25, half %27, i64 1
  %29 = fmul fast half %11, 0xH4200
  %30 = insertelement <4 x half> %28, half %29, i64 2
  %31 = extractelement <4 x half> %8, i64 3
  %32 = fadd fast half %31, 0xH4500
  %33 = insertelement <4 x half> %30, half %32, i64 3
  store <4 x half> %33, ptr addrspace(4) %7, align 8, !tbaa !22
  %34 = extractelement <4 x half> %5, i64 2
  %35 = extractelement <4 x half> %8, i64 2
  %36 = fadd fast half %35, %34
  %37 = fadd fast half %36, %11
  store half %37, ptr addrspace(4) %10, align 8, !tbaa !25
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="32" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(read) }
attributes #2 = { nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_aliased_imageblock_planes, !10, !11}
!10 = !{}
!11 = !{!12, !14}
!12 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 24, !"air.struct_type_info", !13, !"air.alias_implicit_imageblock", !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ColorBlock, layout_explicit>", !"air.arg_name", !"block"}
!13 = !{i32 0, i32 8, i32 0, !"half4", !"color", i32 8, i32 8, i32 0, !"half4", !"aux", i32 16, i32 2, i32 0, !"half", !"depth"}
!14 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"position"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/privatevalidation/fixtures/public/kernel_aliased_imageblock_planes.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26, !27, i64 16}
!26 = !{!"_ZTS10ColorBlock", !23, i64 0, !23, i64 8, !27, i64 16}
!27 = !{!"half", !23, i64 0}
