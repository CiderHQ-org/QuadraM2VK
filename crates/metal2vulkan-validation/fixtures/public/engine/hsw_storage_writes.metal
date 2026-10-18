// Owned fixture (written for this repository): a kernel's `float` write
// to a storage texture, one thread per texel, of buffer 0's element at the thread's index
// (`gid.y * width + gid.x`, the row's width in buffer 1): through `access::write` and
// `access::read_write` of a `texture2d`, through a `texture_buffer`, and through an argument
// buffer's `texture2d` (buffer 2). Bound to a half-float texture, Metal rounds each channel toward
// zero. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void hsw_write(texture2d<float, access::write> t [[texture(0)]],
                      device const float4 *values [[buffer(0)]],
                      constant uint &width [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]])
{
    t.write(values[gid.y * width + gid.x], gid);
}

kernel void hsw_read_write(texture2d<float, access::read_write> t [[texture(0)]],
                           device const float4 *values [[buffer(0)]],
                           constant uint &width [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]])
{
    t.write(values[gid.y * width + gid.x], gid);
}

kernel void hsw_buffer(texture_buffer<float, access::write> t [[texture(0)]],
                       device const float4 *values [[buffer(0)]],
                       uint gid [[thread_position_in_grid]])
{
    t.write(values[gid], gid);
}

struct HswHeld {
    texture2d<float, access::write> t;
};

kernel void hsw_held(constant HswHeld &held [[buffer(2)]],
                     device const float4 *values [[buffer(0)]],
                     constant uint &width [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]])
{
    held.t.write(values[gid.y * width + gid.x], gid);
}
