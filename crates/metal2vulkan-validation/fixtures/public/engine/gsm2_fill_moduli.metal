// Owned fixture (written for this repository): the quad and
// simdgroup shuffle-and-fill under moduli Metal answers by whether they are constants, in kernels,
// a fragment function and a vertex function, and a vertex function of quad fills, rotation, first
// lane and prefix sums with no quad reduction.
// - Thread or vertex t reads in[t] and fills with in[t] + 1000; the fragment's pixel (x, y) holds
//   x + 64y and fills with 10000 more. The float inputs are 1, -1, 2, -2, 0.5 and -0.5.
// - A constant modulo m fills the simdgroup in segments of the next power of two of m, at least
//   2 (3 in fours, 5 in eights, past the quad), and 33 in pairs. A runtime modulo moves the lanes
//   only when it is a power of two from 2 to the group's width (2 and 4 for a quad, up to 32 for
//   a simdgroup); any other leaves each lane its own data. Buffer 2 holds the runtime
//   (delta, modulo) pairs.
// - The cases run full simdgroups only: a segment of 8 to 32 lanes reads past the quad.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void fill_constant_moduli(const device float *in [[buffer(0)]],
                                 device float *out [[buffer(1)]],
                                 uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    const float y = x + 1000.0f;
    out[12 * t + 0] = quad_shuffle_and_fill_up(x, y, ushort(1), ushort(3));
    out[12 * t + 1] = quad_shuffle_and_fill_down(x, y, ushort(2), ushort(3));
    out[12 * t + 2] = quad_shuffle_and_fill_up(x, y, ushort(3), ushort(5));
    out[12 * t + 3] = quad_shuffle_and_fill_down(x, y, ushort(1), ushort(5));
    out[12 * t + 4] = quad_shuffle_and_fill_up(x, y, ushort(1), ushort(33));
    out[12 * t + 5] = quad_shuffle_and_fill_down(x, y, ushort(1), ushort(0));
    out[12 * t + 6] = simd_shuffle_and_fill_up(x, y, ushort(1), ushort(3));
    out[12 * t + 7] = simd_shuffle_and_fill_down(x, y, ushort(2), ushort(5));
    out[12 * t + 8] = simd_shuffle_and_fill_up(x, y, ushort(5), ushort(33));
    out[12 * t + 9] = simd_shuffle_and_fill_down(x, y, ushort(3), ushort(20));
    out[12 * t + 10] = simd_shuffle_and_fill_up(x, y, ushort(2), ushort(1000));
    out[12 * t + 11] = quad_shuffle_and_fill_down(x, y, ushort(6), ushort(12));
}

kernel void fill_runtime_moduli(const device float *in [[buffer(0)]],
                                device float *out [[buffer(1)]],
                                constant uint *dm [[buffer(2)]],
                                uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    const float y = x + 1000.0f;
    out[8 * t + 0] = quad_shuffle_and_fill_up(x, y, ushort(dm[0]), ushort(dm[1]));
    out[8 * t + 1] = simd_shuffle_and_fill_up(x, y, ushort(dm[0]), ushort(dm[1]));
    out[8 * t + 2] = quad_shuffle_and_fill_down(x, y, ushort(dm[2]), ushort(dm[3]));
    out[8 * t + 3] = simd_shuffle_and_fill_down(x, y, ushort(dm[2]), ushort(dm[3]));
    out[8 * t + 4] = quad_shuffle_and_fill_up(x, y, ushort(dm[4]), ushort(dm[5]));
    out[8 * t + 5] = simd_shuffle_and_fill_down(x, y, ushort(dm[4]), ushort(dm[5]));
    out[8 * t + 6] = quad_shuffle_and_fill_down(x, y, ushort(dm[6]), ushort(dm[7]));
    out[8 * t + 7] = simd_shuffle_and_fill_up(x, y, ushort(dm[6]), ushort(dm[7]));
}

// A modulo the AIR gives as a function constant, which Metal specializes when it makes the
// pipeline: the translation takes it as a runtime modulo (the open part).
constant ushort fill_modulo [[function_constant(0)]];

kernel void fill_function_constant_modulo(const device float *in [[buffer(0)]],
                                          device float *out [[buffer(1)]],
                                          uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    const float y = x + 1000.0f;
    out[2 * t + 0] = quad_shuffle_and_fill_up(x, y, ushort(1), fill_modulo);
    out[2 * t + 1] = simd_shuffle_and_fill_down(x, y, ushort(2), fill_modulo);
}

fragment float4 fragment_fill_constant_moduli(float4 position [[position]])
{
    const uint2 p = uint2(position.xy);
    const float x = float(p.x + 64u * p.y);
    const float y = x + 10000.0f;
    return float4(quad_shuffle_and_fill_up(x, y, ushort(1), ushort(3)),
                  quad_shuffle_and_fill_down(x, y, ushort(1), ushort(5)),
                  simd_shuffle_and_fill_up(x, y, ushort(2), ushort(3)),
                  simd_shuffle_and_fill_down(x, y, ushort(1), ushort(33)));
}

vertex void fill_constant_moduli_in_a_vertex_function(uint vid [[vertex_id]],
                                                      const device float *in [[buffer(0)]],
                                                      device float *out [[buffer(1)]])
{
    const float x = in[vid];
    const float y = x + 1000.0f;
    out[2 * vid + 0] = quad_shuffle_and_fill_up(x, y, ushort(1), ushort(3));
    out[2 * vid + 1] = simd_shuffle_and_fill_down(x, y, ushort(3), ushort(5));
}

vertex void quad_fills_in_a_vertex_function(uint vid [[vertex_id]],
                                            const device float *in [[buffer(0)]],
                                            device float *out [[buffer(1)]])
{
    const float x = in[vid];
    const float y = x + 1000.0f;
    out[6 * vid + 0] = quad_shuffle_and_fill_up(x, y, ushort(1));
    out[6 * vid + 1] = quad_shuffle_and_fill_down(x, y, ushort(1), ushort(2));
    out[6 * vid + 2] = quad_shuffle_rotate_up(x, ushort(1));
    out[6 * vid + 3] = quad_broadcast_first(x);
    out[6 * vid + 4] = quad_prefix_exclusive_sum(x);
    out[6 * vid + 5] = quad_prefix_inclusive_product(x);
}
