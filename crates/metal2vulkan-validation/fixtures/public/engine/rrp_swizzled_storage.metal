// Owned fixture: reads, writes and an atomic
// through the texture at [[texture(0)]], which a case binds as a swizzled view, or as a texture
// whose declared usage the access skips. A read kernel writes texel (0, 0) to buffer 0; a write
// kernel writes buffer 1's value to its thread's texel; the atomic kernel adds 5 to texel (0, 0)
// and writes the value it held to buffer 0. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void rrp_sw_read_f(texture2d<float, access::read> v [[texture(0)]],
                          device float4 *o [[buffer(0)]])
{
    o[0] = v.read(uint2(0, 0));
}

kernel void rrp_sw_rwread_f(texture2d<float, access::read_write> v [[texture(0)]],
                            device float4 *o [[buffer(0)]])
{
    o[0] = v.read(uint2(0, 0));
}

kernel void rrp_sw_write_f(texture2d<float, access::write> v [[texture(0)]],
                           constant float4 *value [[buffer(1)]],
                           uint2 g [[thread_position_in_grid]])
{
    v.write(value[0], g);
}

kernel void rrp_sw_rwwrite_f(texture2d<float, access::read_write> v [[texture(0)]],
                             constant float4 *value [[buffer(1)]],
                             uint2 g [[thread_position_in_grid]])
{
    v.write(value[0], g);
}

kernel void rrp_sw_rwread_u(texture2d<uint, access::read_write> v [[texture(0)]],
                            device uint4 *o [[buffer(0)]])
{
    o[0] = v.read(uint2(0, 0));
}

kernel void rrp_sw_write_u(texture2d<uint, access::write> v [[texture(0)]],
                           constant uint4 *value [[buffer(1)]],
                           uint2 g [[thread_position_in_grid]])
{
    v.write(value[0], g);
}

kernel void rrp_sw_atomic_u(texture2d<uint, access::read_write> v [[texture(0)]],
                            device uint4 *o [[buffer(0)]])
{
    o[0] = v.atomic_fetch_add(uint2(0, 0), uint4(5u));
}
