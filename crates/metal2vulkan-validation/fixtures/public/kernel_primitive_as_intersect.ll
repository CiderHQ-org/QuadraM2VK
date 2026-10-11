; Owned synthetic fixture for the AIR primitive triangle-intersection ABI.
; Not derived from a third-party metallib.
source_filename = "kernel_primitive_as_intersect.metal"

define void @primitive_as_intersect(ptr addrspace(1) %acceleration_structure, ptr addrspace(1) %output) {
  %table = call ptr addrspace(1) @air.get_null_intersection_function_table()
  %hit = call { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.triangle_data(<3 x float> <float 0.0, float 0.0, float 1.0>, <3 x float> <float 0.0, float 0.0, float -1.0>, float 0.0, float 10.0, ptr addrspace(1) %acceleration_structure, ptr addrspace(1) %table, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 -1, i32 -1, i32 0, i1 false)
  %type = extractvalue { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } %hit, 0
  %distance = extractvalue { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } %hit, 1
  %primitive_id = extractvalue { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } %hit, 2
  %barycentrics = extractvalue { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } %hit, 5
  %front = extractvalue { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } %hit, 6
  store i32 %type, ptr addrspace(1) %output, align 4
  %distance_slot = getelementptr inbounds i32, ptr addrspace(1) %output, i64 1
  store float %distance, ptr addrspace(1) %distance_slot, align 4
  %primitive_slot = getelementptr inbounds i32, ptr addrspace(1) %output, i64 2
  store i32 %primitive_id, ptr addrspace(1) %primitive_slot, align 4
  %bary_x = extractelement <2 x float> %barycentrics, i64 0
  %bary_x_slot = getelementptr inbounds i32, ptr addrspace(1) %output, i64 3
  store float %bary_x, ptr addrspace(1) %bary_x_slot, align 4
  %bary_y = extractelement <2 x float> %barycentrics, i64 1
  %bary_y_slot = getelementptr inbounds i32, ptr addrspace(1) %output, i64 4
  store float %bary_y, ptr addrspace(1) %bary_y_slot, align 4
  %front32 = zext i1 %front to i32
  %front_slot = getelementptr inbounds i32, ptr addrspace(1) %output, i64 5
  store i32 %front32, ptr addrspace(1) %front_slot, align 4
  ret void
}

declare ptr addrspace(1) @air.get_null_intersection_function_table()
declare { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.triangle_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @primitive_as_intersect, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.primitive_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<>", !"air.arg_name", !"acceleration_structure"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
