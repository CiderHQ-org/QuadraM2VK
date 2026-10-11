#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

kernel void primitive_as_intersect(primitive_acceleration_structure acceleration_structure [[buffer(5)]],
                                   device uint *output [[buffer(0)]]) {
    ray query;
    query.origin = float3(0.0f, 0.0f, 1.0f);
    query.direction = float3(0.0f, 0.0f, -1.0f);
    query.min_distance = 0.0f;
    query.max_distance = 10.0f;

    intersector<triangle_data> trace;
    auto hit = trace.intersect(query, acceleration_structure);
    output[0] = uint(hit.type);
    output[1] = as_type<uint>(hit.distance);
    output[2] = hit.primitive_id;
    output[3] = as_type<uint>(hit.triangle_barycentric_coord.x);
    output[4] = as_type<uint>(hit.triangle_barycentric_coord.y);
    output[5] = hit.triangle_front_facing ? 1u : 0u;
}
