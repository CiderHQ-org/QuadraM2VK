// metal-std: metal4.0
// metal-macos-min: 26.2
// Owned fixture: matmul2d's float sums where the whole-tile readings
// (tns_mm_float) do not reach: k, m, and n cut by the tensors' extents, inner extents past the
// descriptor's k, denormal, infinite, NaN, and overflowing values, relaxed precision, multiply mode,
// and tiles of two and four simdgroups. Threadgroup t computes one tile: A (extents a0, a1) over
// a + t * a0 * a1 and B (extents b0, b1) over b + t * b0 * b1, in device memory or (where the
// variant says) as a cooperative right input filled by coordinate (zero past B's extents); C, a
// cooperative float destination, starts from c + t * block by coordinate (row stride 64), and out
// + t * block receives it by coordinate. Function constant 0 (0 when absent) selects the
// descriptor, the execution scope, the element type of A and B, and where B lives. Loops bounded by
// the capacity.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint selected [[function_constant(0)]];
// Without the constant, variant 0.
constant uint variant = is_function_constant_defined(selected) ? selected : 0;

struct Shape { int a0, a1, b0, b1; uint block; };

using Mode = matmul2d_descriptor::mode;
constexpr constant matmul2d_descriptor nn =
    matmul2d_descriptor(16, 16, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt =
    matmul2d_descriptor(16, 16, 32, false, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn_relaxed =
    matmul2d_descriptor(16, 16, 32, false, false, true, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn_multiply =
    matmul2d_descriptor(16, 16, 32, false, false, false, Mode::multiply);
constexpr constant matmul2d_descriptor nn16x64 =
    matmul2d_descriptor(16, 64, 16, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn64 =
    matmul2d_descriptor(64, 64, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt64 =
    matmul2d_descriptor(64, 64, 32, false, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt32 =
    matmul2d_descriptor(32, 32, 32, false, true, false, Mode::multiply_accumulate);

constant uint OUT_STRIDE = 64;

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
            ct[i] = c[idx[1] * OUT_STRIDE + idx[0]];
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
            out[idx[1] * OUT_STRIDE + idx[0]] = ct[i];
        }
    }
}

kernel void tns_mm_edge(device const uchar *a [[buffer(0)]],
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
    case 0: RUN(nn, S1, float, false);
    case 1: RUN(nn, S1, half, false);
    case 2: RUN(nn, S1, bfloat, false);
    case 3: RUN(nn_relaxed, S1, float, false);
    case 4: RUN(nn_relaxed, S1, half, false);
    case 5: RUN(nn_relaxed, S1, bfloat, false);
    case 6: RUN(nn_multiply, S1, float, false);
    case 7: RUN(nt, S1, float, false);
    case 8: RUN(nn, S1, float, true);
    case 9: RUN(nn16x64, S2, float, false);
    case 10: RUN(nn64, S4, half, false);
    case 11: RUN(nn64, S4, float, false);
    case 12: RUN(nt32, S1, bfloat, false);
    default: RUN(nt64, S4, half, false);
    }
#undef RUN
}
