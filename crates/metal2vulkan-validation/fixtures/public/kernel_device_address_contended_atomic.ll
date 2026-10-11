; Hand edit (DESIGN D149): the `llvm.lifetime.*` calls and declarations take the two
; arguments LLVM 21's `llvm-dis` writes (`i64 <alloca size in bytes>, ptr`), which LLVM 22+
; auto-upgrades by dropping the size, so every gate's `llvm-as` accepts the text.
; ModuleID = 'k.air'
source_filename = "validation/fixtures/public/kernel_device_address_contended_atomic.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.cache = type { ptr addrspace(1), ptr addrspace(1) }

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_device_address_contended_atomic(ptr addrspace(1) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = alloca %struct.cache, align 8
  %5 = bitcast ptr %4 to ptr
  call void @llvm.lifetime.start.p0(i64 16, ptr %4)
  %6 = getelementptr inbounds %struct.cache, ptr %4, i64 0, i32 0
  store ptr addrspace(1) %0, ptr %6, align 8, !tbaa !22
  %7 = getelementptr inbounds %struct.cache, ptr %4, i64 0, i32 1
  store ptr addrspace(1) %1, ptr %7, align 8, !tbaa !27
  %8 = and i32 %2, 3
  call fastcc void @_ZL4bumpRK5cachej(ptr noundef nonnull align 8 dereferenceable(16) %4, i32 noundef %8) #4
  %9 = zext i32 %2 to i64
  %10 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %9
  store i32 %8, ptr addrspace(1) %10, align 4, !tbaa !28, !alias.scope !30, !noalias !33
  call void @llvm.lifetime.end.p0(i64 16, ptr %4)
  ret void
}

; Function Attrs: mustprogress noinline nounwind willreturn
define internal fastcc void @_ZL4bumpRK5cachej(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(16) %0, i32 noundef %1) unnamed_addr #1 {
  %3 = getelementptr inbounds %struct.cache, ptr %0, i64 0, i32 0
  %4 = load ptr addrspace(1), ptr %3, align 8, !tbaa !22
  %5 = zext i32 %1 to i64
  %6 = getelementptr inbounds float, ptr addrspace(1) %4, i64 %5
  %7 = bitcast ptr addrspace(1) %6 to ptr addrspace(1)
  %8 = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none) %7, i32 1, i32 0, i32 2, i1 true) #5
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none), i32, i32, i32, i1) local_unnamed_addr #2

declare void @llvm.lifetime.start.i64(i64)

declare void @llvm.lifetime.end.i64(i64)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #3

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress noinline nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { mustprogress nounwind willreturn }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nobuiltin "no-builtins" }
attributes #5 = { nounwind willreturn }

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
!9 = !{ptr @kernel_device_address_contended_atomic, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"slots"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"seen"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/private/tmp/mkfix/kernel_device_address_contended_atomic.metal"}
!22 = !{!23, !24, i64 0}
!23 = !{!"_ZTS5cache", !24, i64 0, !24, i64 8}
!24 = !{!"any pointer", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!23, !24, i64 8}
!28 = !{!29, !29, i64 0}
!29 = !{!"int", !25, i64 0}
!30 = !{!31}
!31 = distinct !{!31, !32, !"air-alias-scope-arg(1)"}
!32 = distinct !{!32, !"air-alias-scopes(kernel_device_address_contended_atomic)"}
!33 = !{!34}
!34 = distinct !{!34, !32, !"air-alias-scope-arg(0)"}
