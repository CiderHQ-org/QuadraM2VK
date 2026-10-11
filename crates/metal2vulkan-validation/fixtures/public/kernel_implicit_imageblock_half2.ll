define void @kernel_implicit_imageblock_half2(ptr addrspace(4) %block, <2 x i16> %position) {
entry:
  %value = call <2 x half> @air.load.implicit_imageblock.v2f16(i32 0, <2 x i16> %position, i32 0, i16 0)
  call void @air.store.implicit_imageblock.v2f16(<2 x half> %value, i32 0, <2 x i16> %position, i32 0, i16 0)
  ret void
}

declare <2 x half> @air.load.implicit_imageblock.v2f16(i32, <2 x i16>, i32, i16)
declare void @air.store.implicit_imageblock.v2f16(<2 x half>, i32, <2 x i16>, i32, i16)

!air.kernel = !{!0}
!0 = !{ptr @kernel_implicit_imageblock_half2, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.imageblock", !"implicit", !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"imageblock<ImplicitColor, layout_implicit>"}
!4 = !{i32 0, i32 4, i32 0, !"half2", !"value", !"air.render_target", i32 0}
!5 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2"}
