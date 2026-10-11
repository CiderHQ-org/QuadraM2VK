; Hand edit (DESIGN D149): the `llvm.lifetime.*` calls and declarations take the two
; arguments LLVM 21's `llvm-dis` writes (`i64 <alloca size in bytes>, ptr`), which LLVM 22+
; auto-upgrades by dropping the size, so every gate's `llvm-as` accepts the text.
; ModuleID = 'f.fix.ll'
source_filename = "f.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

%"struct.metal::array" = type { [3 x %"struct.metal::texture2d"] }
%"struct.metal::texture2d" = type { ptr addrspace(1) }

; Function Attrs: convergent mustprogress nofree nounwind willreturn
define void @kernel_sampled_texture_array_select(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(2) readonly captures(none) %3, ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %4, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %5, i32 %6) local_unnamed_addr #0 {
  %8 = alloca %"struct.metal::array", align 8
  %9 = bitcast ptr %8 to ptr
  call void @llvm.lifetime.start.p0(i64 24, ptr %8)
  %10 = getelementptr inbounds %"struct.metal::array", ptr %8, i64 0, i32 0, i64 0, i32 0
  store ptr addrspace(1) %0, ptr %10, align 8
  %11 = getelementptr inbounds %"struct.metal::array", ptr %8, i64 0, i32 0, i64 1, i32 0
  store ptr addrspace(1) %1, ptr %11, align 8
  %12 = getelementptr inbounds %"struct.metal::array", ptr %8, i64 0, i32 0, i64 2, i32 0
  store ptr addrspace(1) %2, ptr %12, align 8
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %13
  %15 = load i32, ptr addrspace(1) %14, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %16 = urem i32 %15, 3
  %17 = zext i32 %16 to i64
  %18 = getelementptr inbounds %"struct.metal::array", ptr %8, i64 0, i32 0, i64 %17, i32 0
  %19 = load ptr addrspace(1), ptr %18, align 8, !tbaa !37
  %20 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %19, ptr addrspace(2) readonly captures(none) %3, <2 x float> <float 2.500000e-01, float 7.500000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #3
  %21 = extractvalue { <4 x float>, i8 } %20, 0
  %22 = extractelement <4 x float> %21, i64 0
  %23 = extractelement <4 x float> %21, i64 1
  %24 = fmul fast float %23, 1.000000e+01
  %25 = fadd fast float %24, %22
  %26 = extractelement <4 x float> %21, i64 2
  %27 = fmul fast float %26, 1.000000e+02
  %28 = fadd fast float %25, %27
  %29 = extractelement <4 x float> %21, i64 3
  %30 = fmul fast float %29, 1.000000e+03
  %31 = fadd fast float %28, %30
  %32 = getelementptr inbounds float, ptr addrspace(1) %5, i64 %13
  store float %31, ptr addrspace(1) %32, align 4, !tbaa !40, !alias.scope !42, !noalias !43
  call void @llvm.lifetime.end.p0(i64 24, ptr %8)
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr #1

declare void @llvm.lifetime.start.i64(i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #2

declare void @llvm.lifetime.end.i64(i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #2

attributes #0 = { convergent mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { convergent nounwind willreturn memory(argmem: read) }

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
!9 = !{ptr @kernel_sampled_texture_array_select, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t0"}
!13 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t1"}
!14 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t2"}
!15 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"samp"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"which"}
!17 = !{i32 5, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!18 = !{i32 6, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 3, i32 0, i32 0}
!25 = !{!"/private/tmp/imgsel/f.metal"}
!26 = !{!27, !27, i64 0}
!27 = !{!"int", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"air-alias-scope-arg(4)"}
!32 = distinct !{!32, !"air-alias-scopes(kernel_sampled_texture_array_select)"}
!33 = !{!34, !35, !36}
!34 = distinct !{!34, !32, !"air-alias-scope-textures"}
!35 = distinct !{!35, !32, !"air-alias-scope-samplers"}
!36 = distinct !{!36, !32, !"air-alias-scope-arg(5)"}
!37 = !{!38, !39, i64 0}
!38 = !{!"_ZTSN5metal9texture2dIfLNS_6accessE0EvEE", !39, i64 0}
!39 = !{!"__metal_texture_2d_t", !28, i64 0}
!40 = !{!41, !41, i64 0}
!41 = !{!"float", !28, i64 0}
!42 = !{!36}
!43 = !{!34, !35, !31}
