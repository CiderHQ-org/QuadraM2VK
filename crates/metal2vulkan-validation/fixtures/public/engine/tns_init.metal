// metal-std: metal4.0
// metal-macos-min: 26.2
// Owned fixture: what constructing a cooperative tensor leaves in its
// storage. Function constant 0 (0 when absent) selects a descriptor, an execution scope, element
// types, and an operand (the destination, or the right input). For each of `rounds` rounds (a
// buffer, so the loop stays a loop), each thread constructs the operand's cooperative tensor in the
// loop's body, as MetalPerformanceShaders' gather kernels construct their destinations, and writes
// the bits of each element i it holds (at most 64) to out[(round * threads + tid) * 64 + i]; then
// it sets element i to round * 32 + i + 1 and writes the sum of its elements, read back, to
// out[rounds * threads * 64 + round * threads + tid]. Round 0 reads a first construction, every
// later round a construction over the storage the round before set. Loops bounded by the capacity.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint selected [[function_constant(0)]];
// Without the constant, variant 0.
constant uint variant = is_function_constant_defined(selected) ? selected : 0;

using Mode = matmul2d_descriptor::mode;

constant uint MAX_HELD = 64;

static uint bits(float v) { return as_type<uint>(v); }
static uint bits(half v) { return uint(as_type<ushort>(v)); }
static uint bits(int v) { return uint(v); }
static uint bits(int8_t v) { return uint(uchar(v)); }

template <typename CT>
static void round_of(thread CT &ct, device uint *out, uint tid, uint threads, uint r,
                     uint rounds) {
    using E = typename CT::element_type;
    uint cap = min(uint(ct.get_capacity()), MAX_HELD);
    device uint *held = out + (r * threads + tid) * MAX_HELD;
    for (uint i = 0; i < cap; ++i)
        held[i] = bits(E(ct[i]));
    for (uint i = 0; i < cap; ++i)
        ct[i] = E(r * 32 + i + 1);
    uint sum = 0;
    for (uint i = 0; i < cap; ++i)
        sum += uint(E(ct[i]));
    out[rounds * threads * MAX_HELD + r * threads + tid] = sum;
}

template <matmul2d_descriptor D, typename S, typename L, typename R, typename T>
static void destination(device uint *out, uint tid, uint threads, uint rounds) {
    matmul2d<D, S> op;
    using LT = tensor<device L, dextents<int32_t, 2>, tensor_inline>;
    using RT = tensor<device R, dextents<int32_t, 2>, tensor_inline>;
    for (uint r = 0; r < rounds; ++r) {
        auto ct = op.template get_destination_cooperative_tensor<LT, RT, T>();
        round_of(ct, out, tid, threads, r, rounds);
    }
}

template <matmul2d_descriptor D, typename S, typename L, typename R, typename T>
static void right(device uint *out, uint tid, uint threads, uint rounds) {
    matmul2d<D, S> op;
    for (uint r = 0; r < rounds; ++r) {
        auto ct = op.template get_right_input_cooperative_tensor<L, R, T>();
        round_of(ct, out, tid, threads, r, rounds);
    }
}

constexpr constant matmul2d_descriptor nn16 =
    matmul2d_descriptor(16, 16, 32, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn16x64 =
    matmul2d_descriptor(16, 64, 16, false, false, false, Mode::multiply_accumulate);
constexpr constant matmul2d_descriptor nn64 =
    matmul2d_descriptor(64, 64, 32, false, false, false, Mode::multiply_accumulate);

kernel void tns_init(device uint *out [[buffer(0)]],
                     constant uint &rounds [[buffer(1)]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]]) {
    using S1 = execution_simdgroup;
    using S2 = execution_simdgroups<2>;
    using S4 = execution_simdgroups<4>;
    switch (variant) {
    case 0: destination<nn16, S1, half, half, float>(out, tid, threads, rounds); break;
    case 1: destination<nn16, S1, float, float, float>(out, tid, threads, rounds); break;
    case 2: destination<nn16, S1, int8_t, int8_t, int>(out, tid, threads, rounds); break;
    case 3: destination<nn16x64, S2, half, half, float>(out, tid, threads, rounds); break;
    case 4: destination<nn64, S4, half, half, float>(out, tid, threads, rounds); break;
    case 5: right<nn16, S1, half, half, float>(out, tid, threads, rounds); break;
    case 6: right<nn16, S1, float, float, float>(out, tid, threads, rounds); break;
    default: right<nn16, S1, int8_t, int8_t, int>(out, tid, threads, rounds); break;
    }
}
