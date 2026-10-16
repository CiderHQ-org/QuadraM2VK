// Sixteen screen coordinates mapped to physical coordinates through one of two rasterization-rate
// maps, picked from a thread-memory array of their pointers.
//
// Both maps are buffer arguments, `left` at buffer 0 and `right` at buffer 1. The kernel stores
// their addresses in a local `maps[2]` array and maps through `*maps[tid.y]`; the compiler keeps
// the array (an `alloca` of two pointers, a store of each buffer, and a load at the dynamic index),
// the shape a corpus row picks its map by. Every pointer the array holds is a whole buffer at
// offset 0, so a case authors both buffers as `rasterization_rate_map`, and each executor writes
// its own implementation's parameter data of the same maps.
//
// The maps are the two whose tiles this crate compiles as Apple M3 does: `left` is
// a 64x64 map of uniform quality 1.0 (the identity), and `right` is 37x23 with horizontal samples
// [0.5, 1] and vertical samples [1, 0.5] (physical size 36x23). The eight coordinates lie inside
// both screens, at exact binary fractions; row y = 0 answers through `left` (each coordinate
// itself) and row y = 1 through `right`, whose x scale moves seven of them.
#include <metal_stdlib>
using namespace metal;

constant float2 kScreen[8] = {
    float2(0.0f, 0.0f),
    float2(1.5f, 2.5f),
    float2(10.25f, 7.75f),
    float2(17.5f, 11.5f),
    float2(20.125f, 3.875f),
    float2(31.25f, 20.75f),
    float2(35.5f, 21.5f),
    float2(5.0f, 18.0f),
};

kernel void kernel_rate_map_pointer_array(
    constant rasterization_rate_map_data& left [[buffer(0)]],
    constant rasterization_rate_map_data& right [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]])
{
    constant rasterization_rate_map_data* maps[2] = { &left, &right };
    rasterization_rate_map_decoder decoder(*maps[tid.y]);
    float2 physical = decoder.map_screen_to_physical_coordinates(kScreen[tid.x]);
    uint lane = tid.y * 8 + tid.x;
    out[lane * 2 + 0] = as_type<uint>(physical.x);
    out[lane * 2 + 1] = as_type<uint>(physical.y);
}
