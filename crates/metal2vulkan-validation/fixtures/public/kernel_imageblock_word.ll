; Owned synthetic fixture for the authored compute-imageblock execution contract.
; Not derived from a third-party metallib.
source_filename = "kernel_imageblock_word.metal"

%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @imageblock_word(%"struct.metal::_imageblock_base" %block, ptr addrspace(1) %output, <2 x i16> %thread_position) {
  %cell = call ptr addrspace(4) @air.imageblock_data(<2 x i16> %thread_position, i32 0, i16 0)
  store i32 42, ptr addrspace(4) %cell, align 4
  %value = load i32, ptr addrspace(4) %cell, align 4
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)

!air.kernel = !{!0}
!0 = !{ptr @imageblock_word, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 4, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"imageblock<ImageblockWord, layout_explicit>", !"air.arg_name", !"block"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"value"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
!6 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"thread_position"}
