// Owned fixture: a kernel's `[[stage_in]]` attributes, fetched under its
// compute pipeline's stage-input descriptor.
//
// The descriptor, not the source, chooses each attribute's format, buffer, offset, and step
// (constant, thread position in the grid on x or y at a step rate, or through the index buffer),
// so one kernel exercises every step: the tests fetch known bytes under each descriptor and
// compare the bits each thread reports with the element Metal's step names.
#include <metal_stdlib>
using namespace metal;

struct Fetched {
    float4 f [[attribute(0)]];
    uint4 u [[attribute(1)]];
    int i [[attribute(2)]];
    half2 h [[attribute(3)]];
};

// Thread t = y * width + x reports every attribute's bits:
//     out[3t] = bits(f), out[3t + 1] = u, out[3t + 2] = (bits(i), bits(h.x), bits(h.y), t)
kernel void stage_in_fetch(
    Fetched in [[stage_in]],
    device uint4 *out [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]])
{
    uint t = gid.y * grid.x + gid.x;
    out[3 * t] = as_type<uint4>(in.f);
    out[3 * t + 1] = in.u;
    out[3 * t + 2] = uint4(as_type<uint>(in.i), as_type<ushort>(in.h.x), as_type<ushort>(in.h.y), t);
}

struct Sample {
    float v [[attribute(0)]];
};

// The stage-in region beside a fetched attribute:
//     out[2t] = (bits(v), origin), out[2t + 1] = (size, t)
kernel void stage_in_region(
    Sample in [[stage_in]],
    device uint4 *out [[buffer(0)]],
    uint t [[thread_position_in_grid]],
    uint3 origin [[grid_origin]],
    uint3 size [[grid_size]])
{
    out[2 * t] = uint4(as_type<uint>(in.v), origin);
    out[2 * t + 1] = uint4(size, t);
}

// The region's origin and size as 16-bit pairs, without a stage input:
//     out[t] = (origin, size)
kernel void stage_in_region_short(
    device ushort4 *out [[buffer(0)]],
    uint t [[thread_position_in_grid]],
    ushort2 origin [[grid_origin]],
    ushort2 size [[grid_size]])
{
    out[t] = ushort4(origin, size);
}
