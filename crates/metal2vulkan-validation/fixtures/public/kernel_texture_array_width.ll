%"struct.metal::texture2d" = type { ptr addrspace(1) }

define void @texture_array_width(ptr readonly captures(none) %textures, ptr addrspace(1) noundef writeonly captures(none) %out) local_unnamed_addr #0 {
entry:
  %element = getelementptr %"struct.metal::texture2d", ptr %textures, i64 1, i32 0
  %handle = load ptr addrspace(1), ptr %element, align 8
  %width = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none) %handle, i32 0) #1
  store i32 %width, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #1

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nounwind memory(none) }

!air.kernel = !{!0}
!0 = !{ptr @texture_array_width, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 2, !"air.sample", !"air.arg_type_name", !"array<texture2d<float, sample>, 2>", !"air.arg_name", !"textures"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
