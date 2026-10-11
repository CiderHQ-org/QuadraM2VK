#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

kernel void instance_as_world_space(
    instance_acceleration_structure acceleration_structure [[buffer(5)]],
    intersection_function_table<instancing, world_space_data> intersection_table [[buffer(6)]],
    device uint *output [[buffer(0)]]) {
    ray query;
    query.origin = float3(0.0f, 0.0f, 1.0f);
    query.direction = float3(0.0f, 0.0f, -1.0f);
    query.min_distance = 0.0f;
    query.max_distance = 10.0f;

    intersector<instancing, world_space_data> trace;
    auto hit = trace.intersect(query, acceleration_structure, 0xffu, intersection_table);
    output[0] = uint(hit.type);
    output[1] = as_type<uint>(hit.distance);
    constexpr float3 weights(1.0f, 2.0f, 4.0f);
    for (uint column = 0; column < 4; ++column) {
        output[2 + column] = as_type<uint>(dot(hit.world_to_object_transform[column], weights));
        output[6 + column] = as_type<uint>(dot(hit.object_to_world_transform[column], weights));
    }
}
