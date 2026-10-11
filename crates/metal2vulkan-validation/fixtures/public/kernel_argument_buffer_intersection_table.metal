#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct RayResources {
    intersection_function_table<instancing, triangle_data> table [[id(0)]];
};

kernel void argument_buffer_intersection_table(
    instance_acceleration_structure acceleration_structure [[buffer(5)]],
    constant RayResources &resources [[buffer(6)]],
    device uint *output [[buffer(0)]]) {
    ray query;
    query.origin = float3(0.0f, 0.0f, 1.0f);
    query.direction = float3(0.0f, 0.0f, -1.0f);
    query.min_distance = 0.0f;
    query.max_distance = 10.0f;

    intersector<instancing, triangle_data> trace;
    auto hit = trace.intersect(query, acceleration_structure, 0xffu, resources.table);
    output[0] = uint(hit.type);
}
