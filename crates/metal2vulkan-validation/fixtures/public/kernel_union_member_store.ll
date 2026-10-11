; Generated from kernel_union_member_store.metal by
;   xcrun metal -std=metal3.0 -S -emit-llvm, retripled to spirv-unknown-vulkan1.2, opt -S.
; One hand edit (DESIGN D149): the `llvm.lifetime.*` calls and declarations take the two
; arguments LLVM 21's `llvm-dis` writes (`i64 <alloca size in bytes>, ptr`), which LLVM 22+
; auto-upgrades by dropping the size, so every gate's `llvm-as` accepts the text.
;
; The shape this reaches: two stores whose pointer's declared pointee is `%union.Slot`, the
; one-member LLVM struct an MSL union lowers to. `store float` (a narrowing member store) and
; `store <2 x i32>` (a same-width one) both used to fall through every width-based rule --
; `bitcast_width` has no answer for a struct -- and the module was refused with
; "owned Store violates its pointer-pointee and value-type contract: ... points at %6 (TypeStruct
; %5), but the stored value is ... TypeFloat 32". `emit_single_member_struct_store` descends to
; member 0, which is the slot's own address, and the two ordinary rules take it from there.
;
; The DYNAMIC read index `r`, taken from a second element of the input buffer, is load-bearing:
; read back through `j` or `other` and the frontend forwards both member stores to the load, so
; neither is observed and both are dead.
source_filename = "v3.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

%union.Slot = type { i64 }

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_union_member_store(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, i32 %2) local_unnamed_addr #0 {
  %4 = alloca [2 x %union.Slot], align 8
  %5 = bitcast ptr %4 to ptr
  call void @llvm.lifetime.start.p0(i64 16, ptr %4)
  %6 = zext i32 %2 to i64
  %7 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %6
  %8 = load i32, ptr addrspace(1) %7, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %9 = and i32 %8, 1
  %10 = xor i32 %9, 1
  %11 = zext i32 %9 to i64
  %12 = getelementptr inbounds [2 x %union.Slot], ptr %4, i64 0, i64 %11
  %13 = getelementptr %union.Slot, ptr %12, i64 0, i32 0
  store i64 -2401053089206453570, ptr %13, align 8, !tbaa !31
  %14 = zext i32 %10 to i64
  %15 = getelementptr inbounds [2 x %union.Slot], ptr %4, i64 0, i64 %14
  %16 = tail call fast float @air.convert.f.f32.u.i32(i32 %8) #3
  %17 = fadd fast float %16, 5.000000e-01
  %18 = bitcast ptr %12 to ptr
  store float %17, ptr %18, align 8, !tbaa !31
  %19 = add i32 %8, 3
  %20 = insertelement <2 x i32> <i32 poison, i32 -17958194>, i32 %19, i64 0
  %21 = bitcast ptr %15 to ptr
  store <2 x i32> %20, ptr %21, align 8, !tbaa !31
  %22 = add i32 %2, 8
  %23 = zext i32 %22 to i64
  %24 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %23
  %25 = load i32, ptr addrspace(1) %24, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %26 = and i32 %25, 1
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds [2 x %union.Slot], ptr %4, i64 0, i64 %27, i32 0
  %29 = load i64, ptr %28, align 8, !tbaa !31
  %30 = trunc i64 %29 to i32
  %31 = shl i32 %2, 1
  %32 = zext i32 %31 to i64
  %33 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %32
  store i32 %30, ptr addrspace(1) %33, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %34 = lshr i64 %29, 32
  %35 = trunc i64 %34 to i32
  %36 = or i32 %31, 1
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %37
  store i32 %35, ptr addrspace(1) %38, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  call void @llvm.lifetime.end.p0(i64 16, ptr %4)
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

declare void @llvm.lifetime.start.i64(i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #2

declare void @llvm.lifetime.end.i64(i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #2

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_union_member_store, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 3, i32 0, i32 0}
!21 = !{!"/private/tmp/uc/v3.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_union_member_store)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = !{!24, !24, i64 0}
