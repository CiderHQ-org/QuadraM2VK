// metal-std: metal4.0
// metal-macos-min: 26.2
// Owned fixture: matmul2d over int8_t into int32_t, whose sums are exact. One
// threadgroup: A (extents a0, a1) and B (extents b0, b1) are int8_t tensors over a and b (copied to
// threadgroup memory where the variant places them there), C is a cooperative destination
// initialized from c0 by coordinate, and out receives it by coordinate (row stride 64). Function
// constant 0 (0 when absent) selects the descriptor (transposes, multiply or multiply_accumulate, a
// 16 x 16 x 32, 32 x 32 x 32, or two-simdgroup 16 x 64 x 16 tile) and where each operand lives
// (device, threadgroup, or a cooperative right input). The cases' extents give whole tiles, partial
// tiles, and an inner extent other than the descriptor's k. Loops bounded by the capacity and the
// extents.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint selected [[function_constant(0)]];
// Without the constant, variant 0.
constant uint variant = is_function_constant_defined(selected) ? selected : 0;

struct Extents { int a0, a1, b0, b1; };

using Mode = matmul2d_descriptor::mode;
constexpr constant matmul2d_descriptor nn =
    matmul2d_descriptor(16, 16, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt =
    matmul2d_descriptor(16, 16, 32, false, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor tn =
    matmul2d_descriptor(16, 16, 32, true, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor tt =
    matmul2d_descriptor(16, 16, 32, true, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn_multiply =
    matmul2d_descriptor(16, 16, 32, false, false, false, Mode::multiply);
constexpr constant matmul2d_descriptor nt32 =
    matmul2d_descriptor(32, 32, 32, false, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn16x64 =
    matmul2d_descriptor(16, 64, 16, false, false, false, Mode::multiply_accumulate);

enum Where { Device, Threadgroup, Cooperative };

constant uint OUT_STRIDE = 64;

template <matmul2d_descriptor D, typename S, Where L, Where R>
static void product(device int8_t *a, device int8_t *b, device const int *c0, device int *out,
                    constant Extents &e, threadgroup int8_t *ta, threadgroup int8_t *tb,
                    uint tid, uint threads) {
    using DT = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
    using TT = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;
    dextents<int32_t, 2> ae(e.a0, e.a1);
    dextents<int32_t, 2> be(e.b0, e.b1);
    if (L == Threadgroup || R == Threadgroup) {
        for (int i = tid; i < 2048; i += threads) {
            if (L == Threadgroup && i < e.a0 * e.a1) ta[i] = a[i];
            if (R == Threadgroup && i < e.b0 * e.b1) tb[i] = b[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    DT dA(a, ae);
    DT dB(b, be);
    TT tA(ta, ae);
    TT tB(tb, be);
    matmul2d<D, S> op;
    auto ct = op.template get_destination_cooperative_tensor<DT, DT, int>();
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            ct[i] = c0[idx[1] * OUT_STRIDE + idx[0]];
        }
    }
    if constexpr (R == Cooperative) {
        auto rt = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int>();
        for (ushort i = 0; i < rt.get_capacity(); ++i) {
            if (rt.is_valid_element(i)) {
                auto idx = rt.get_multidimensional_index(i);
                rt[i] = (idx[0] < e.b0 && idx[1] < e.b1) ? b[idx[1] * e.b0 + idx[0]] : int8_t(0);
            }
        }
        op.run(dA, rt, ct);
    } else if constexpr (L == Device && R == Device) {
        op.run(dA, dB, ct);
    } else if constexpr (L == Device) {
        op.run(dA, tB, ct);
    } else if constexpr (R == Device) {
        op.run(tA, dB, ct);
    } else {
        op.run(tA, tB, ct);
    }
    for (ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            out[idx[1] * OUT_STRIDE + idx[0]] = ct[i];
        }
    }
}

kernel void tns_mm_int(device const int8_t *a [[buffer(0)]],
                       device const int8_t *b [[buffer(1)]],
                       device const int *c0 [[buffer(2)]],
                       device int *out [[buffer(3)]],
                       constant Extents &e [[buffer(4)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]]) {
    threadgroup int8_t ta[2048];
    threadgroup int8_t tb[2048];
    device int8_t *ma = const_cast<device int8_t *>(a);
    device int8_t *mb = const_cast<device int8_t *>(b);
    using S1 = execution_simdgroup;
    using S2 = execution_simdgroups<2>;
#define RUN(D, S, L, R) product<D, S, L, R>(ma, mb, c0, out, e, ta, tb, tid, threads); break
    switch (variant) {
    case 0: RUN(nn, S1, Device, Device);
    case 1: RUN(nt, S1, Device, Device);
    case 2: RUN(tn, S1, Device, Device);
    case 3: RUN(tt, S1, Device, Device);
    case 4: RUN(nn_multiply, S1, Device, Device);
    case 5: RUN(nn, S1, Device, Threadgroup);
    case 6: RUN(nn, S1, Threadgroup, Threadgroup);
    case 7: RUN(nn, S1, Threadgroup, Device);
    case 8: RUN(nn, S1, Device, Cooperative);
    case 9: RUN(nt32, S1, Device, Device);
    default: RUN(nn16x64, S2, Device, Device);
    }
#undef RUN
}
