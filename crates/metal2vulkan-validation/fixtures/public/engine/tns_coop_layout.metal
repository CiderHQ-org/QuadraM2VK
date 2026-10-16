// metal-std: metal4.0
// metal-macos-min: 26.2
// Owned fixture: where MetalPerformancePrimitives' matmul2d places a
// cooperative tensor's elements. Function constant 0 (0 when absent) selects a descriptor, an
// execution scope, element types, and an operand (the destination, or the right input). Each thread
// writes the tensor's capacity to out[tid] and, for each element i it holds, one word at
// out[threads + tid * capacity + i] (while that index is below `limit`): bit 31 when the element is
// valid, bits 16 to 30 its byte offset from element 0 (0x7FFF when no slot below the capacity holds
// it), and for a valid element its coordinates, the first in bits 0 to 7 and the second in bits 8 to
// 15. The offset is found without pointer arithmetic on addresses: every element i is set to i + 1,
// and the storage, read as an array of the element type from element 0, is searched for that value.
// The layout is the implementation's (the header says so), but a guest can address elements by
// offset from element 0, so a translation must place them as Metal does. Loops bounded by the
// capacity.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint selected [[function_constant(0)]];
// Without the constant, variant 0.
constant uint variant = is_function_constant_defined(selected) ? selected : 0;

using Mode = matmul2d_descriptor::mode;

template <typename CT>
static void record(thread CT &ct, device uint *out, uint tid, uint threads, uint limit) {
    using E = typename CT::element_type;
    uint cap = ct.get_capacity();
    out[tid] = cap;
    for (uint i = 0; i < cap; ++i)
        ct[i] = E(i + 1);
    thread E *base = &ct[0];
    for (uint i = 0; i < cap; ++i) {
        uint at = tid * cap + i;
        if (at >= limit)
            break;
        uint offset = 0x7FFFu;
        for (uint slot = 0; slot < cap; ++slot) {
            if (base[slot] == E(i + 1)) {
                offset = slot * uint(sizeof(E));
                break;
            }
        }
        uint word = offset << 16;
        if (ct.is_valid_element(i)) {
            auto idx = ct.get_multidimensional_index(i);
            word |= 0x80000000u | (uint(idx[1]) & 0xFFu) << 8 | (uint(idx[0]) & 0xFFu);
        }
        out[threads + at] = word;
    }
}

template <matmul2d_descriptor D, typename S, typename L, typename R, typename T>
static void destination(device uint *out, uint tid, uint threads, uint limit) {
    matmul2d<D, S> op;
    using LT = tensor<device L, dextents<int32_t, 2>, tensor_inline>;
    using RT = tensor<device R, dextents<int32_t, 2>, tensor_inline>;
    auto ct = op.template get_destination_cooperative_tensor<LT, RT, T>();
    record(ct, out, tid, threads, limit);
}

template <matmul2d_descriptor D, typename S, typename L, typename R, typename T>
static void right(device uint *out, uint tid, uint threads, uint limit) {
    matmul2d<D, S> op;
    auto ct = op.template get_right_input_cooperative_tensor<L, R, T>();
    record(ct, out, tid, threads, limit);
}

constexpr constant matmul2d_descriptor nn16 =
    matmul2d_descriptor(16, 16, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt16 =
    matmul2d_descriptor(16, 16, 32, false, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn32 =
    matmul2d_descriptor(32, 32, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nt32 =
    matmul2d_descriptor(32, 32, 32, false, true, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn16x64 =
    matmul2d_descriptor(16, 64, 16, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn128 =
    matmul2d_descriptor(128, 128, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn8x128 =
    matmul2d_descriptor(8, 128, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn64 =
    matmul2d_descriptor(64, 64, 32, false, false, false, Mode::multiply_accumulate);

kernel void tns_coop_layout(device uint *out [[buffer(0)]],
                            constant uint &limit [[buffer(1)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]]) {
    using S1 = execution_simdgroup;
    using S2 = execution_simdgroups<2>;
    using S4 = execution_simdgroups<4>;
    switch (variant) {
    case 0: destination<nn16, S1, float, float, float>(out, tid, threads, limit); break;
    case 1: destination<nn16, S1, half, half, float>(out, tid, threads, limit); break;
    case 2: destination<nn16, S1, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 3: destination<nt16, S1, float, float, float>(out, tid, threads, limit); break;
    case 4: destination<nn32, S1, float, float, float>(out, tid, threads, limit); break;
    case 5: right<nn16, S1, float, float, float>(out, tid, threads, limit); break;
    case 6: right<nn16, S1, half, half, float>(out, tid, threads, limit); break;
    case 7: right<nn16, S1, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 8: right<nt16, S1, float, float, float>(out, tid, threads, limit); break;
    case 9: right<nt16, S1, half, half, float>(out, tid, threads, limit); break;
    case 10: right<nt16, S1, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 11: right<nn32, S1, float, float, float>(out, tid, threads, limit); break;
    case 12: right<nn32, S1, half, half, float>(out, tid, threads, limit); break;
    case 13: right<nn32, S1, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 14: right<nt32, S1, float, float, float>(out, tid, threads, limit); break;
    case 15: right<nt32, S1, half, half, float>(out, tid, threads, limit); break;
    case 16: right<nt32, S1, bfloat, bfloat, float>(out, tid, threads, limit); break;
    case 17: destination<nn16x64, S2, float, float, float>(out, tid, threads, limit); break;
    case 18: destination<nn128, S4, half, half, float>(out, tid, threads, limit); break;
    case 19: destination<nn8x128, S4, half, half, float>(out, tid, threads, limit); break;
    case 20: destination<nn64, S4, half, half, float>(out, tid, threads, limit); break;
    case 21: destination<nn16, S1, int8_t, int8_t, int>(out, tid, threads, limit); break;
    case 22: right<nn16, S1, int8_t, int8_t, int>(out, tid, threads, limit); break;
    default: destination<nn16x64, S2, int8_t, int8_t, int>(out, tid, threads, limit); break;
    }
}
