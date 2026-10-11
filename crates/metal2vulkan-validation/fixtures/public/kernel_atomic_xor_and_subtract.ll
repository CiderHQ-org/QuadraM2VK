; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = '/tmp/alias/at.bc'
source_filename = "validation/fixtures/public/kernel_atomic_xor_and_subtract.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%"struct.metal::_atomic" = type { i32 }
%"struct.metal::_atomic.0" = type { i32 }
%"struct.metal::_atomic.1" = type { float }

@_ZZ30kernel_atomic_xor_and_subtractPU9MTLdeviceN5metal7_atomicIjvEEPU9MTLdeviceNS0_IivEEPU9MTLdeviceNS0_IfvEEjE10tile_total = internal addrspace(3) global %"struct.metal::_atomic" zeroinitializer, align 4

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_atomic_xor_and_subtract(ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %4
  tail call void @air.atomic.local.store.i32(ptr addrspace(3) captures(none) @_ZZ30kernel_atomic_xor_and_subtractPU9MTLdeviceN5metal7_atomicIjvEEPU9MTLdeviceNS0_IivEEPU9MTLdeviceNS0_IfvEEjE10tile_total, i32 0, i32 0, i32 1, i1 true) #4
  br label %7

7:                                                ; preds = %6, %4
  tail call void @air.wg.barrier(i32 2, i32 1) #5
  %8 = mul i32 %3, -1640531535
  %9 = getelementptr inbounds %"struct.metal::_atomic", ptr addrspace(1) %0, i64 0, i32 0
  %10 = tail call i32 @air.atomic.global.xor.u.i32(ptr addrspace(1) captures(none) %9, i32 %8, i32 0, i32 2, i1 true) #4
  %11 = add nsw i32 %3, 1
  %12 = getelementptr inbounds %"struct.metal::_atomic.0", ptr addrspace(1) %1, i64 0, i32 0
  %13 = tail call i32 @air.atomic.global.sub.s.i32(ptr addrspace(1) captures(none) %12, i32 %11, i32 0, i32 2, i1 true) #4
  %14 = tail call fast float @air.convert.f.f32.u.i32(i32 %11) #6
  %15 = getelementptr inbounds %"struct.metal::_atomic.1", ptr addrspace(1) %2, i64 0, i32 0
  %16 = tail call fast float @air.atomic.global.sub.f32(ptr addrspace(1) captures(none) %15, float %14, i32 0, i32 2, i1 true) #4
  %17 = tail call i32 @air.atomic.local.sub.u.i32(ptr addrspace(3) captures(none) @_ZZ30kernel_atomic_xor_and_subtractPU9MTLdeviceN5metal7_atomicIjvEEPU9MTLdeviceNS0_IivEEPU9MTLdeviceNS0_IfvEEjE10tile_total, i32 %11, i32 0, i32 1, i1 true) #4
  tail call void @air.wg.barrier(i32 2, i32 1) #5
  br i1 %5, label %18, label %21

18:                                               ; preds = %7
  %19 = getelementptr inbounds %"struct.metal::_atomic", ptr addrspace(1) %0, i64 1, i32 0
  %20 = tail call i32 @air.atomic.local.load.i32(ptr addrspace(3) captures(none) @_ZZ30kernel_atomic_xor_and_subtractPU9MTLdeviceN5metal7_atomicIjvEEPU9MTLdeviceNS0_IivEEPU9MTLdeviceNS0_IfvEEjE10tile_total, i32 0, i32 1, i1 true) #4
  tail call void @air.atomic.global.store.i32(ptr addrspace(1) captures(none) %19, i32 %20, i32 0, i32 2, i1 true) #4
  br label %21

21:                                               ; preds = %18, %7
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn
declare void @air.atomic.local.store.i32(ptr addrspace(3) captures(none), i32, i32, i32, i1) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.global.xor.u.i32(ptr addrspace(1) captures(none), i32, i32, i32, i1) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.global.sub.s.i32(ptr addrspace(1) captures(none), i32, i32, i32, i1) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn
declare float @air.atomic.global.sub.f32(ptr addrspace(1) captures(none), float, i32, i32, i1) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.local.sub.u.i32(ptr addrspace(3) captures(none), i32, i32, i32, i1) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.local.load.i32(ptr addrspace(3) captures(none), i32, i32, i1) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn
declare void @air.atomic.global.store.i32(ptr addrspace(1) captures(none), i32, i32, i32, i1) local_unnamed_addr #3

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent mustprogress nounwind willreturn }
attributes #3 = { mustprogress nounwind willreturn }
attributes #4 = { nounwind willreturn }
attributes #5 = { convergent nounwind willreturn }
attributes #6 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_atomic_xor_and_subtract, !10, !11}
!10 = !{}
!11 = !{!12, !14, !16, !18}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !13, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"unsigned_words"}
!13 = !{i32 0, i32 4, i32 0, !"uint", !"__s"}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !15, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"signed_total"}
!15 = !{i32 0, i32 4, i32 0, !"int", !"__s"}
!16 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !17, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"float_total"}
!17 = !{i32 0, i32 4, i32 0, !"float", !"__s"}
!18 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 4, i32 0, i32 0}
!25 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_atomic_xor_and_subtract.metal"}
