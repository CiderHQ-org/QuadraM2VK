; Generated from kernel_thread_local_narrowing_store.metal by
;   xcrun metal -std=metal3.0 -S -emit-llvm, retripled to spirv-unknown-vulkan1.2, opt -S.
; One hand edit (DESIGN D149): the `llvm.lifetime.*` calls and declarations take the two
; arguments LLVM 21's `llvm-dis` writes (`i64 <alloca size in bytes>, ptr`), which LLVM 22+
; auto-upgrades by dropping the size, so every gate's `llvm-as` accepts the text.
;
; The shape this reaches: `store float` through a pointer whose declared pointee is a 64-bit
; thread-local slot. Logical SPIR-V has no partial store, so
; `emitter::body::vector_store::emit_scalar_narrowing_store` turns it into a read-modify-write that
; changes only the low four bytes. The DYNAMIC index is load-bearing -- a scalar local is
; scalar-replaced by the frontend and the store never reaches the emitter.
;
; Spelling it as an MSL `union { ulong u; float f; }` does NOT translate: the union lowers to a
; one-member LLVM struct, so the store's pointee is a `TypeStruct` rather than the scalar the rule
; requires, and the module is refused with `owned Store violates its pointer-pointee and value-type
; contract`. The reinterpreting cast keeps the pointee scalar.
; ModuleID = 'g.fix.ll'
source_filename = "g.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_thread_local_narrowing_store(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, i32 %2) local_unnamed_addr #0 {
  %4 = alloca [2 x i64], align 8
  %5 = bitcast ptr %4 to ptr
  call void @llvm.lifetime.start.p0(i64 16, ptr %4)
  %6 = zext i32 %2 to i64
  %7 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %6
  %8 = load i32, ptr addrspace(1) %7, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %9 = and i32 %8, 1
  %10 = xor i32 %9, 1
  %11 = zext i32 %9 to i64
  %12 = getelementptr inbounds [2 x i64], ptr %4, i64 0, i64 %11
  store i64 -2401053089206453570, ptr %12, align 8, !tbaa !31
  %13 = zext i32 %10 to i64
  %14 = getelementptr inbounds [2 x i64], ptr %4, i64 0, i64 %13
  store i64 81985529216486895, ptr %14, align 8, !tbaa !31
  %15 = tail call fast float @air.convert.f.f32.u.i32(i32 %8) #3
  %16 = fadd fast float %15, 5.000000e-01
  %17 = bitcast ptr %12 to ptr
  store float %16, ptr %17, align 8, !tbaa !33
  %18 = load i64, ptr %12, align 8, !tbaa !31
  %19 = trunc i64 %18 to i32
  %20 = mul i32 %2, 3
  %21 = zext i32 %20 to i64
  %22 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %21
  store i32 %19, ptr addrspace(1) %22, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %23 = lshr i64 %18, 32
  %24 = trunc i64 %23 to i32
  %25 = add i32 %20, 1
  %26 = zext i32 %25 to i64
  %27 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %26
  store i32 %24, ptr addrspace(1) %27, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %28 = add i32 %20, 2
  %29 = zext i32 %28 to i64
  %30 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %29
  store i32 -1985229329, ptr addrspace(1) %30, align 4, !tbaa !22, !alias.scope !29, !noalias !26
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
!9 = !{ptr @kernel_thread_local_narrowing_store, !10, !11}
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
!21 = !{!"/private/tmp/fx6/g.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_thread_local_narrowing_store)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = !{!32, !32, i64 0}
!32 = !{!"long", !24, i64 0}
!33 = !{!34, !34, i64 0}
!34 = !{!"float", !24, i64 0}
