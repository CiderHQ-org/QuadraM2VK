// Owned fixture: the instance count of an instance acceleration structure,
// which no Vulkan instruction reads: the binder supplies the bound structure's count, so the
// kernel translates without ray queries. out[gid] = count * 3 + gid. Loop-free.
#include <metal_stdlib>
using namespace metal;
using namespace metal::raytracing;

kernel void rt_instance_count(device uint *out [[buffer(0)]],
                              instance_acceleration_structure scene [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
    out[gid] = scene.get_instance_count() * 3u + gid;
}
