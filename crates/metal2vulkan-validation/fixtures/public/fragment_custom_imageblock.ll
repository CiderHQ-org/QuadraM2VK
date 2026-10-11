target triple = "spirv-unknown-vulkan1.3"
%Depth = type { half }

define { <4 x half>, %Depth } @fragment_custom_imageblock(%Depth %input) {
entry:
  %depth = extractvalue %Depth %input, 0
  %next = fadd half %depth, 0xH3C00
  %color = insertelement <4 x half> zeroinitializer, half %depth, i32 0
  %projection = insertvalue %Depth poison, half %next, 0
  %out0 = insertvalue { <4 x half>, %Depth } poison, <4 x half> %color, 0
  %out1 = insertvalue { <4 x half>, %Depth } %out0, %Depth %projection, 1
  ret { <4 x half>, %Depth } %out1
}

!air.fragment = !{!0}
!0 = !{ptr @fragment_custom_imageblock, !1, !2}
!1 = !{!3, !4}
!2 = !{!5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{!"air.imageblock_data", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !6, !"air.imageblock_master", !7}
!5 = !{i32 0, !"air.imageblock_data", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !6, !"air.imageblock_master", !7}
!6 = !{i32 0, i32 2, i32 0, !"half", !"user(depth)"}
!7 = !{i32 0, i32 2, i32 0, !"half", !"user(warped_type)", !"air.raster_order_group", i32 0, i32 2, i32 2, i32 0, !"half", !"user(depth)", !"air.raster_order_group", i32 0, i32 4, i32 2, i32 0, !"half", !"user(blending_weight)", !"air.raster_order_group", i32 0, i32 6, i32 2, i32 0, !"half", !"user(depth_buffer)", !"air.raster_order_group", i32 0}
