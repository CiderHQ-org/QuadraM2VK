// Owned fixture: an array of textures indexed dynamically
// (`access.handle.array_load.clamped`: the index into the sized handle array is clamped under
// isolation). Thread g reads texel (g >> 2, 0) of texture g & 3: out[g] = ts[g & 3](g >> 2).
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_texture_array_dynamic(array<texture2d<float>, 4> ts [[texture(0)]],
                                     device float4 *out [[buffer(0)]],
                                     uint gid [[thread_position_in_grid]]) {
    out[gid] = ts[gid & 3].read(uint2(gid >> 2, 0));
}
