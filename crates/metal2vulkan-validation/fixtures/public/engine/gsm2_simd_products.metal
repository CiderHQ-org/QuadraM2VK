// Owned fixture (written for this repository): simdgroup products, prefix
// products and rotate up, in kernels and in vertex functions. Thread or vertex t reads in[t]. The
// float and half inputs are 1, -1, 2, -2, 0.5 and -0.5, at most eight of magnitude 2 and six of
// 0.5 in a simdgroup, so every product, prefix product and partial product of any association
// order is exact; the integer inputs are odd, so no product is zero, and integer products wrap.
// simd_shuffle_rotate_up reads lanes of whole simdgroups only (its entries' cases run full
// simdgroups). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void simd_float_products(const device float *in [[buffer(0)]],
                                device float *out [[buffer(1)]],
                                uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    out[4 * t + 0] = simd_product(x);
    out[4 * t + 1] = simd_prefix_inclusive_product(x);
    out[4 * t + 2] = simd_prefix_exclusive_product(x);
    out[4 * t + 3] = simd_product(float2(x, -x)).y;
}

kernel void simd_half_products(const device half *in [[buffer(0)]],
                               device half *out [[buffer(1)]],
                               uint t [[thread_position_in_grid]])
{
    const half x = in[t];
    out[4 * t + 0] = simd_product(x);
    out[4 * t + 1] = simd_prefix_inclusive_product(x);
    out[4 * t + 2] = simd_prefix_exclusive_product(x);
    out[4 * t + 3] = simd_prefix_inclusive_product(half2(x, -x)).y;
}

kernel void simd_int_products(const device int *in [[buffer(0)]],
                              device uint *out [[buffer(1)]],
                              uint t [[thread_position_in_grid]])
{
    const int x = in[t];
    out[7 * t + 0] = uint(simd_product(x));
    out[7 * t + 1] = uint(simd_prefix_inclusive_product(x));
    out[7 * t + 2] = uint(simd_prefix_exclusive_product(x));
    out[7 * t + 3] = simd_product(uint(x) * 3u);
    out[7 * t + 4] = uint(simd_product(ushort(x)));
    out[7 * t + 5] = uint(simd_prefix_inclusive_product(uchar(x)));
    out[7 * t + 6] = uint(int(simd_prefix_exclusive_product(short(x))));
}

kernel void simd_products_divergent(const device float *in [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    float r = -7.0f;
    float s = -7.0f;
    if (t % 3 != 0) {
        r = simd_product(x);
        s = simd_prefix_exclusive_product(x);
    }
    out[2 * t + 0] = r;
    out[2 * t + 1] = s;
}

kernel void simd_rotate_up(const device float *in [[buffer(0)]],
                           device float *out [[buffer(1)]],
                           constant uint &delta [[buffer(2)]],
                           uint t [[thread_position_in_grid]])
{
    const float x = in[t];
    out[4 * t + 0] = simd_shuffle_rotate_up(x, ushort(3));
    out[4 * t + 1] = simd_shuffle_rotate_up(x, ushort(delta));
    out[4 * t + 2] = simd_shuffle_rotate_up(float2(x, -x), ushort(31)).y;
    out[4 * t + 3] = as_type<float>(simd_shuffle_rotate_up(int(t), ushort(1)));
}

vertex void simd_products_in_a_vertex_function(uint vid [[vertex_id]],
                                               const device float *in [[buffer(0)]],
                                               device float *out [[buffer(1)]])
{
    const float x = in[vid];
    out[3 * vid + 0] = simd_product(x);
    out[3 * vid + 1] = simd_prefix_inclusive_product(x);
    out[3 * vid + 2] = simd_prefix_exclusive_product(x);
}

vertex void simd_rotate_up_in_a_vertex_function(uint vid [[vertex_id]],
                                                const device float *in [[buffer(0)]],
                                                device float *out [[buffer(1)]],
                                                constant uint &delta [[buffer(2)]])
{
    const float x = in[vid];
    out[2 * vid + 0] = simd_shuffle_rotate_up(x, ushort(3));
    out[2 * vid + 1] = simd_shuffle_rotate_up(x, ushort(delta));
}
