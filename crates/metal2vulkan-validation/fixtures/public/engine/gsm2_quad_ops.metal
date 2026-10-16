// Owned fixture (written for this repository): the quad group operations
// Metal has beside the simdgroup ones, in kernels, vertex functions and fragment functions, and
// the simdgroup fills in fragment functions.
// - Kernels and vertex functions: thread or vertex t reads in[t]. The float and half inputs are
//   1, -1, 2, -2, 0.5 and -0.5, so every quad product, sum and prefix of any association order is
//   exact; the integer inputs are odd, so no product is zero, and integer products wrap. The
//   fills' filling data is the data plus 1000; buffer 2 holds four (delta, modulo) pairs, the
//   runtime spelling of the fills' arguments. Every modulo is 2 or 4: Metal answers any other
//   differently as a constant and as a runtime value. A partial quad's cases run only the
//   operations that read no lane the quad lacks (a shuffle from one is undefined in Metal).
// - Fragment functions are drawn over the whole target. A pixel's value is the same pattern by
//   its position (`exact_at`), or its index x + 64y for the fills, whose filling data adds 10000.
// - `quad_ops_divergent` runs its operations under a mask of its quad's lanes, its arm a block of
//   several stores, and `fragment_quad_ops_divergent` under a mask of pixels, with operations
//   that read lanes by value alone.
// - `quad_first_in_a_speculated_arm` and `fragment_quad_first_in_a_speculated_arm` are the shape
//   in which Metal's backend speculates the arm: quad_broadcast_first in a branch joined by a phi
//   answers with the quad's lane 0 whether that lane takes the arm or not. The AIR keeps the call
//   in the branch; the translation speculates the arm as Metal does (`air::canon::speculate`).
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void quad_fills(const device float *in [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       constant uint *dm [[buffer(2)]],
                       uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    const float y = x + 1000.0f;
    out[12 * t + 0] = quad_shuffle_and_fill_up(x, y, ushort(1));
    out[12 * t + 1] = quad_shuffle_and_fill_down(x, y, ushort(2));
    out[12 * t + 2] = quad_shuffle_and_fill_up(x, y, ushort(1), ushort(2));
    out[12 * t + 3] = quad_shuffle_and_fill_down(x, y, ushort(3), ushort(4));
    out[12 * t + 4] = quad_shuffle_and_fill_up(x, y, ushort(dm[0]), ushort(dm[1]));
    out[12 * t + 5] = quad_shuffle_and_fill_down(x, y, ushort(dm[2]), ushort(dm[3]));
    out[12 * t + 6] = quad_shuffle_and_fill_up(x, y, ushort(dm[4]), ushort(dm[5]));
    out[12 * t + 7] = quad_shuffle_and_fill_down(x, y, ushort(dm[6]), ushort(dm[7]));
    out[12 * t + 8] = quad_shuffle_and_fill_down(x, y, ushort(dm[0]));
    out[12 * t + 9] = quad_shuffle_rotate_up(x, ushort(1));
    out[12 * t + 10] = quad_shuffle_rotate_up(x, ushort(dm[2]));
    out[12 * t + 11] = quad_shuffle_and_fill_up(float2(x, y), float2(y, x), ushort(2)).y;
}

// The fills of a partial quad read only the lanes it has: fills within lane pairs.
kernel void quad_pair_fills(const device float *in [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    const float y = x + 1000.0f;
    out[4 * t + 0] = quad_shuffle_and_fill_up(x, y, ushort(1), ushort(2));
    out[4 * t + 1] = quad_shuffle_and_fill_down(x, y, ushort(1), ushort(2));
    out[4 * t + 2] = quad_shuffle_and_fill_up(x, y, ushort(2), ushort(2));
    out[4 * t + 3] = quad_shuffle_and_fill_down(x, y, ushort(2), ushort(2));
}

kernel void quad_typed_fills(const device int *in [[buffer(0)]],
                             device uint *out [[buffer(1)]],
                             constant uint *dm [[buffer(2)]],
                             uint t [[thread_position_in_grid]])
{
    const int x = in[t];
    out[4 * t + 0] = uint(quad_shuffle_and_fill_up(x, x + 1000, ushort(dm[0]), ushort(dm[1])));
    out[4 * t + 1] = uint(int(quad_shuffle_and_fill_down(short(x), short(7), ushort(1))));
    out[4 * t + 2] = as_type<uint>(half2(quad_shuffle_and_fill_up(half(x), half(-1), ushort(2), ushort(4)),
                                         quad_shuffle_rotate_up(half(x), ushort(3))));
    out[4 * t + 3] = uint(quad_shuffle_and_fill_down(uchar(x), uchar(9), ushort(1), ushort(2)));
}

kernel void quad_bitwise(const device int *in [[buffer(0)]],
                         device uint *out [[buffer(1)]],
                         uint t [[thread_position_in_grid]])
{
    const int x = in[t];
    out[6 * t + 0] = uint(quad_and(x));
    out[6 * t + 1] = uint(quad_or(x));
    out[6 * t + 2] = quad_xor(uint(x));
    out[6 * t + 3] = uint(quad_or(ushort(x)));
    out[6 * t + 4] = uint(quad_xor(uchar(x)));
    out[6 * t + 5] = quad_broadcast_first(uint(x));
}

kernel void quad_float_scans(const device float *in [[buffer(0)]],
                             device float *out [[buffer(1)]],
                             uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    out[8 * t + 0] = quad_product(x);
    out[8 * t + 1] = quad_sum(x);
    out[8 * t + 2] = quad_prefix_inclusive_sum(x);
    out[8 * t + 3] = quad_prefix_exclusive_sum(x);
    out[8 * t + 4] = quad_prefix_inclusive_product(x);
    out[8 * t + 5] = quad_prefix_exclusive_product(x);
    out[8 * t + 6] = quad_broadcast_first(x);
    out[8 * t + 7] = quad_prefix_exclusive_sum(float2(x, -x)).y;
}

kernel void quad_half_scans(const device half *in [[buffer(0)]],
                            device half *out [[buffer(1)]],
                            uint t [[thread_position_in_grid]])
{
    const half x = in[t];
    out[6 * t + 0] = quad_product(x);
    out[6 * t + 1] = quad_prefix_inclusive_sum(x);
    out[6 * t + 2] = quad_prefix_exclusive_sum(x);
    out[6 * t + 3] = quad_prefix_inclusive_product(x);
    out[6 * t + 4] = quad_prefix_exclusive_product(x);
    out[6 * t + 5] = quad_product(half2(x, -x)).y;
}

kernel void quad_int_scans(const device int *in [[buffer(0)]],
                           device uint *out [[buffer(1)]],
                           uint t [[thread_position_in_grid]])
{
    const int x = in[t];
    out[7 * t + 0] = uint(quad_product(x));
    out[7 * t + 1] = uint(quad_prefix_inclusive_sum(x));
    out[7 * t + 2] = uint(quad_prefix_exclusive_sum(x));
    out[7 * t + 3] = uint(quad_prefix_inclusive_product(x));
    out[7 * t + 4] = quad_prefix_exclusive_product(uint(x) * 3u);
    out[7 * t + 5] = uint(quad_product(ushort(x)));
    out[7 * t + 6] = uint(int(quad_prefix_exclusive_sum(short(x))));
}

kernel void quad_ops_divergent(const device float *in [[buffer(0)]],
                               device float *out [[buffer(1)]],
                               uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    const uint mask = (t / 4) % 16;
    out[6 * t + 0] = -7.0f;
    out[6 * t + 1] = -7.0f;
    out[6 * t + 2] = -7.0f;
    out[6 * t + 3] = -7.0f;
    out[6 * t + 4] = -7.0f;
    out[6 * t + 5] = -7.0f;
    if ((mask >> (t & 3)) & 1) {
        out[6 * t + 0] = quad_broadcast_first(x);
        out[6 * t + 1] = quad_product(x);
        out[6 * t + 2] = quad_prefix_exclusive_sum(x);
        out[6 * t + 3] = quad_prefix_inclusive_product(x);
        out[6 * t + 4] = as_type<float>(quad_or(t << 8) | 0x40000000u);
        out[6 * t + 5] = as_type<float>(quad_xor(t) | 0x40000000u);
    }
}

kernel void quad_first_in_a_speculated_arm(device uint *out [[buffer(1)]],
                                           uint t [[thread_position_in_grid]])
{
    uint r = 7777u;
    if (t % 3 != 0) {
        r = quad_broadcast_first(t * 10u + 1u);
    }
    out[t] = r;
}

vertex void quad_ops_in_a_vertex_function(uint vid [[vertex_id]],
                                          const device float *in [[buffer(0)]],
                                          device float *out [[buffer(1)]],
                                          constant uint *dm [[buffer(2)]])
{
    const float x = in[vid];
    const float y = x + 1000.0f;
    out[8 * vid + 0] = quad_shuffle_and_fill_up(x, y, ushort(1));
    out[8 * vid + 1] = quad_shuffle_and_fill_down(x, y, ushort(dm[0]), ushort(dm[1]));
    out[8 * vid + 2] = quad_shuffle_rotate_up(x, ushort(1));
    out[8 * vid + 3] = quad_broadcast_first(x);
    out[8 * vid + 4] = quad_product(x);
    out[8 * vid + 5] = quad_prefix_exclusive_sum(x);
    out[8 * vid + 6] = quad_prefix_inclusive_product(x);
    out[8 * vid + 7] = as_type<float>(quad_and(vid | 0x40000000u) ^ quad_xor(vid));
}

// The operations a partial quad of vertices answers in full: none reads a lane the quad lacks.
vertex void quad_reductions_in_a_vertex_function(uint vid [[vertex_id]],
                                                 const device float *in [[buffer(0)]],
                                                 device float *out [[buffer(1)]])
{
    const float x = in[vid];
    out[6 * vid + 0] = quad_shuffle_and_fill_up(x, x + 1000.0f, ushort(1), ushort(2));
    out[6 * vid + 1] = quad_broadcast_first(x);
    out[6 * vid + 2] = quad_product(x);
    out[6 * vid + 3] = quad_prefix_exclusive_sum(x);
    out[6 * vid + 4] = quad_prefix_inclusive_product(x);
    out[6 * vid + 5] = as_type<float>(quad_and(vid | 0x40000000u) ^ quad_xor(vid));
}

static uint2 pixel(float4 position)
{
    return uint2(position.xy);
}

static float exact_at(uint2 p)
{
    const uint k = (p.x * 3u + p.y * 5u) % 6u;
    const float m = k < 2u ? 1.0f : (k < 4u ? 2.0f : 0.5f);
    return (k & 1u) != 0u ? -m : m;
}

static float index_at(uint2 p)
{
    return float(p.x + 64u * p.y);
}

fragment float4 fragment_quad_fills(float4 position [[position]])
{
    const float x = index_at(pixel(position));
    const float y = x + 10000.0f;
    return float4(quad_shuffle_and_fill_up(x, y, ushort(1)),
                  quad_shuffle_and_fill_down(x, y, ushort(2)),
                  quad_shuffle_and_fill_up(x, y, ushort(1), ushort(2)),
                  quad_shuffle_and_fill_down(x, y, ushort(1), ushort(2)));
}

fragment float4 fragment_quad_fills_and_rotations(float4 position [[position]])
{
    const float x = index_at(pixel(position));
    const float y = x + 10000.0f;
    return float4(quad_shuffle_and_fill_up(x, y, ushort(3), ushort(4)),
                  quad_shuffle_and_fill_down(x, y, ushort(3), ushort(4)),
                  quad_shuffle_rotate_up(x, ushort(1)),
                  quad_shuffle_rotate_up(x, ushort(3)));
}

fragment float4 fragment_quad_scans(float4 position [[position]])
{
    const float x = exact_at(pixel(position));
    return float4(quad_product(x), quad_prefix_exclusive_sum(x), quad_prefix_inclusive_product(x),
                  quad_broadcast_first(index_at(pixel(position))));
}

fragment uint4 fragment_quad_bitwise(float4 position [[position]])
{
    const uint2 p = pixel(position);
    const uint b = (p.x * 7u + p.y * 13u) ^ 0x5au;
    return uint4(quad_and(b | 0x100u), quad_or(b), quad_xor(b), quad_prefix_inclusive_sum(b));
}

fragment float4 fragment_quad_ops_divergent(float4 position [[position]])
{
    const uint2 p = pixel(position);
    const float x = exact_at(p);
    float4 r = float4(-7.0f);
    if ((p.x + 2u * p.y) % 3u != 0u) {
        r = float4(quad_product(x), quad_prefix_exclusive_sum(x), quad_prefix_inclusive_product(x),
                   as_type<float>(quad_or(p.x << 8) | 0x40000000u));
    }
    return r;
}

fragment float4 fragment_quad_first_in_a_speculated_arm(float4 position [[position]])
{
    const uint2 p = pixel(position);
    float r = -7.0f;
    if ((p.x + 2u * p.y) % 3u != 0u) {
        r = quad_broadcast_first(index_at(p));
    }
    return float4(r, 0.0f, 0.0f, 1.0f);
}

fragment float4 fragment_simd_fills(float4 position [[position]])
{
    const float x = index_at(pixel(position));
    const float y = x + 10000.0f;
    return float4(simd_shuffle_and_fill_up(x, y, ushort(1)),
                  simd_shuffle_and_fill_down(x, y, ushort(3)),
                  simd_shuffle_and_fill_up(x, y, ushort(2), ushort(8)),
                  simd_shuffle_and_fill_down(x, y, ushort(5), ushort(16)));
}
