; Owned synthetic fixture for AIR single-level instance transform results.
; Not derived from a third-party metallib.
source_filename = "kernel_instance_as_world_space.metal"

define void @instance_as_world_space(ptr addrspace(1) %acceleration_structure, ptr addrspace(1) %intersection_table, ptr addrspace(1) %output) {
  %hit = call { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } @air.intersect.instancing.world_space_data(<3 x float> <float 0.0, float 0.0, float 1.0>, <3 x float> <float 0.0, float 0.0, float -1.0>, float 0.0, float 10.0, ptr addrspace(1) %acceleration_structure, i32 255, ptr addrspace(1) %intersection_table, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 -1, i32 -1, i32 0, i1 false, i1 false)
  %type = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 0
  %distance = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 1
  store i32 %type, ptr addrspace(1) %output, align 4
  %distance_slot = getelementptr inbounds i32, ptr addrspace(1) %output, i64 1
  store float %distance, ptr addrspace(1) %distance_slot, align 4
  %world_to_object_0 = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 7
  %world_to_object_1 = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 8
  %world_to_object_2 = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 9
  %world_to_object_3 = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 10
  %object_to_world_0 = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 11
  %object_to_world_1 = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 12
  %object_to_world_2 = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 13
  %object_to_world_3 = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 14
  %w2o0 = call float @air.dot.v3f32(<3 x float> %world_to_object_0, <3 x float> <float 1.0, float 2.0, float 4.0>)
  %w2o1 = call float @air.dot.v3f32(<3 x float> %world_to_object_1, <3 x float> <float 1.0, float 2.0, float 4.0>)
  %w2o2 = call float @air.dot.v3f32(<3 x float> %world_to_object_2, <3 x float> <float 1.0, float 2.0, float 4.0>)
  %w2o3 = call float @air.dot.v3f32(<3 x float> %world_to_object_3, <3 x float> <float 1.0, float 2.0, float 4.0>)
  %o2w0 = call float @air.dot.v3f32(<3 x float> %object_to_world_0, <3 x float> <float 1.0, float 2.0, float 4.0>)
  %o2w1 = call float @air.dot.v3f32(<3 x float> %object_to_world_1, <3 x float> <float 1.0, float 2.0, float 4.0>)
  %o2w2 = call float @air.dot.v3f32(<3 x float> %object_to_world_2, <3 x float> <float 1.0, float 2.0, float 4.0>)
  %o2w3 = call float @air.dot.v3f32(<3 x float> %object_to_world_3, <3 x float> <float 1.0, float 2.0, float 4.0>)
  %slot2 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 2
  store float %w2o0, ptr addrspace(1) %slot2, align 4
  %slot3 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 3
  store float %w2o1, ptr addrspace(1) %slot3, align 4
  %slot4 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 4
  store float %w2o2, ptr addrspace(1) %slot4, align 4
  %slot5 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 5
  store float %w2o3, ptr addrspace(1) %slot5, align 4
  %slot6 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 6
  store float %o2w0, ptr addrspace(1) %slot6, align 4
  %slot7 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 7
  store float %o2w1, ptr addrspace(1) %slot7, align 4
  %slot8 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 8
  store float %o2w2, ptr addrspace(1) %slot8, align 4
  %slot9 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 9
  store float %o2w3, ptr addrspace(1) %slot9, align 4
  ret void
}

declare float @air.dot.v3f32(<3 x float>, <3 x float>)
declare { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } @air.intersect.instancing.world_space_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)

!air.kernel = !{!0}
!0 = !{ptr @instance_as_world_space, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.instance_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing>", !"air.arg_name", !"acceleration_structure"}
!4 = !{i32 1, !"air.intersection_function_table", !"air.location_index", i32 6, i32 1, !"air.read_write", !"air.arg_type_name", !"intersection_function_table<instancing, world_space_data>", !"air.arg_name", !"intersection_table"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 40, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
