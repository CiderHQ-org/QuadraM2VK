// metal-std: metal4.0
// metal-macos-min: 26.2
// Owned fixture: matmul2d's float sums with relaxed precision and in
// multiply mode over the shapes, transposes, element types, and scopes the plain sums were read
// over (tns_mm_float, tns_mm_edge), and the plain sums over the two shapes only the layout probe
// read (128 x 128 x 32 and 8 x 128 x 32 over four simdgroups). Threadgroup t computes one tile: A
// (extents a0, a1) over a + t * a0 * a1 and B (extents b0, b1) over b + t * b0 * b1, in device
// memory or (where the variant says) as a cooperative right input filled by coordinate (zero past
// B's extents); C, a cooperative float destination, starts from c + t * block by coordinate (row
// stride `stride`), and out + t * block receives it by coordinate. Function constant 0 (0 when
// absent) selects the descriptor, the execution scope, the element type of A and B, and where B
// lives. Loops bounded by the capacity.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint selected [[function_constant(0)]];
// Without the constant, variant 0.
constant uint variant = is_function_constant_defined(selected) ? selected : 0;

struct Shape { int a0, a1, b0, b1; uint block, stride; };

using Mode = matmul2d_descriptor::mode;
constexpr constant Mode MA = Mode::multiply_accumulate;
constexpr constant Mode MU = Mode::multiply;
// Relaxed precision, multiply_accumulate.
constexpr constant matmul2d_descriptor r_nn = matmul2d_descriptor(16, 16, 32, false, false, true, MA);
constexpr constant matmul2d_descriptor r_nt = matmul2d_descriptor(16, 16, 32, false, true, true, MA);
constexpr constant matmul2d_descriptor r_16x64 = matmul2d_descriptor(16, 64, 16, false, false, true, MA);
constexpr constant matmul2d_descriptor r_64 = matmul2d_descriptor(64, 64, 32, false, false, true, MA);
constexpr constant matmul2d_descriptor r_64t = matmul2d_descriptor(64, 64, 32, false, true, true, MA);
constexpr constant matmul2d_descriptor r_32t = matmul2d_descriptor(32, 32, 32, false, true, true, MA);
constexpr constant matmul2d_descriptor r_128 = matmul2d_descriptor(128, 128, 32, false, false, true, MA);
constexpr constant matmul2d_descriptor r_8x128 = matmul2d_descriptor(8, 128, 32, false, false, true, MA);
// Multiply mode.
constexpr constant matmul2d_descriptor x_nn = matmul2d_descriptor(16, 16, 32, false, false, false, MU);
constexpr constant matmul2d_descriptor x_nt = matmul2d_descriptor(16, 16, 32, false, true, false, MU);
constexpr constant matmul2d_descriptor x_16x64 = matmul2d_descriptor(16, 64, 16, false, false, false, MU);
constexpr constant matmul2d_descriptor x_64 = matmul2d_descriptor(64, 64, 32, false, false, false, MU);
constexpr constant matmul2d_descriptor x_64t = matmul2d_descriptor(64, 64, 32, false, true, false, MU);
constexpr constant matmul2d_descriptor x_32t = matmul2d_descriptor(32, 32, 32, false, true, false, MU);
constexpr constant matmul2d_descriptor x_128 = matmul2d_descriptor(128, 128, 32, false, false, false, MU);
constexpr constant matmul2d_descriptor x_8x128 = matmul2d_descriptor(8, 128, 32, false, false, false, MU);
// Plain: multiply_accumulate without relaxed precision.
constexpr constant matmul2d_descriptor p_128 = matmul2d_descriptor(128, 128, 32, false, false, false, MA);
constexpr constant matmul2d_descriptor p_8x128 = matmul2d_descriptor(8, 128, 32, false, false, false, MA);

template <matmul2d_descriptor D, typename S, typename E, bool Cooperative>
static void product(device const uchar *a_bytes, device const uchar *b_bytes,
                    device const float *c, device float *out, constant Shape &s, uint t) {
    using DT = tensor<device E, dextents<int32_t, 2>, tensor_inline>;
    device E *a = const_cast<device E *>((device const E *)a_bytes) + t * uint(s.a0 * s.a1);
    device E *b = const_cast<device E *>((device const E *)b_bytes) + t * uint(s.b0 * s.b1);
    c += t * s.block;
    out += t * s.block;
    DT A(a, dextents<int32_t, 2>(s.a0, s.a1));
    matmul2d<D, S> op;
    auto ct = op.template get_destination_cooperative_tensor<DT, DT, float>();
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            ct[i] = c[idx[1] * s.stride + idx[0]];
        }
    }
    if constexpr (Cooperative) {
        auto rt = op.template get_right_input_cooperative_tensor<E, E, float>();
        for (ushort i = 0; i < rt.get_capacity(); ++i) {
            if (rt.is_valid_element(i)) {
                auto idx = rt.get_multidimensional_index(i);
                rt[i] = (idx[0] < s.b0 && idx[1] < s.b1) ? b[idx[1] * s.b0 + idx[0]] : E(0);
            }
        }
        op.run(A, rt, ct);
    } else {
        DT B(b, dextents<int32_t, 2>(s.b0, s.b1));
        op.run(A, B, ct);
    }
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            out[idx[1] * s.stride + idx[0]] = ct[i];
        }
    }
}

kernel void tns_mm_flags(device const uchar *a [[buffer(0)]],
                         device const uchar *b [[buffer(1)]],
                         device const float *c [[buffer(2)]],
                         device float *out [[buffer(3)]],
                         constant Shape &s [[buffer(4)]],
                         uint t [[threadgroup_position_in_grid]]) {
    using S1 = execution_simdgroup;
    using S2 = execution_simdgroups<2>;
    using S4 = execution_simdgroups<4>;
#define RUN(D, S, E, C) product<D, S, E, C>(a, b, c, out, s, t); break
    switch (variant) {
    case 0: RUN(r_nt, S1, float, false);
    case 1: RUN(r_nn, S1, float, true);
    case 2: RUN(r_16x64, S2, float, false);
    case 3: RUN(r_64, S4, half, false);
    case 4: RUN(r_64, S4, float, false);
    case 5: RUN(r_32t, S1, bfloat, false);
    case 6: RUN(r_64t, S4, half, false);
    case 7: RUN(r_128, S4, float, false);
    case 8: RUN(r_8x128, S4, float, false);
    case 9: RUN(x_nt, S1, float, false);
    case 10: RUN(x_nn, S1, float, true);
    case 11: RUN(x_16x64, S2, float, false);
    case 12: RUN(x_64, S4, half, false);
    case 13: RUN(x_64, S4, float, false);
    case 14: RUN(x_32t, S1, bfloat, false);
    case 15: RUN(x_64t, S4, half, false);
    case 16: RUN(x_128, S4, float, false);
    case 17: RUN(x_8x128, S4, float, false);
    case 18: RUN(x_nn, S1, half, false);
    case 19: RUN(x_nn, S1, bfloat, false);
    case 20: RUN(p_128, S4, float, false);
    default: RUN(p_8x128, S4, float, false);
    }
#undef RUN
}
