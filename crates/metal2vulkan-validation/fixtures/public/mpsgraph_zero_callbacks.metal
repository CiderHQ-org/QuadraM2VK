#include <metal_stdlib>
using namespace metal;

struct MPSComplexF32 {
    float real;
    float imaginary;
};

struct MPSKernelInOutPrefix {
    metal::array_ref<device uchar *> buffers;
    uchar unused[40];
};

[[visible]] float prefixPrimary_f(thread uchar *, constant uchar *, uint4, uint) { return 1.0f; }
[[visible]] float prefixSecondary_f(thread uchar *, constant uchar *, uint4, uint) { return 0.0f; }
[[visible]] float prefixTertiary_f(thread uchar *, constant uchar *, uint4, uint) { return 0.0f; }
[[visible]] float prefixQuaternary_f(thread uchar *, constant uchar *, uint4, uint) { return 0.0f; }
[[visible]] float prefixQuinary_f(thread uchar *, constant uchar *, uint4, uint) { return 0.0f; }
[[visible]] float prefixSenary_f(thread uchar *, constant uchar *, uint4, uint) { return 0.0f; }
[[visible]] float prefixSeptenary_f(thread uchar *, constant uchar *, uint4, uint) { return 0.0f; }
[[visible]] float prefixOctonary_f(thread uchar *, constant uchar *, uint4, uint) { return 0.0f; }

[[visible]] float4 prefixPrimary_4xf(thread uchar *, constant uchar *, uint4, uint) { return 0.0f; }
[[visible]] float4 prefixSecondary_4xf(thread uchar *, constant uchar *, uint4, uint) { return 0.0f; }
[[visible]] MPSComplexF32 prefixPrimary_cf(thread uchar *, constant uchar *, uint4, uint) {
    return {1.0f, 2.0f};
}
[[visible]] MPSComplexF32 prefixSecondary_cf(thread uchar *, constant uchar *, uint4, uint) {
    return {0.0f, 0.0f};
}
[[visible]] int prefixPrimary_i(thread uchar *, constant uchar *, uint4, uint) { return 1; }
[[visible]] int prefixSecondary_i(thread uchar *, constant uchar *, uint4, uint) { return 0; }
[[visible]] int prefixTertiary_i(thread uchar *, constant uchar *, uint4, uint) { return 0; }
[[visible]] long prefixPrimary_i64(thread uchar *, constant uchar *, uint4, uint) { return 1; }
[[visible]] uint prefixPrimary_u(thread uchar *, constant uchar *, uint4, uint) { return 1; }
[[visible]] ushort prefixPrimary_u16(thread uchar *, constant uchar *, uint4, uint) { return 0; }
[[visible]] ulong prefixPrimary_u64(thread uchar *, constant uchar *, uint4, uint) { return 1; }
[[visible]] uchar prefixPrimary_u8(thread uchar *, constant uchar *, uint4, uint) { return 0; }

[[visible]] void middlefixPrimary_f(
    thread uchar *, thread uchar *, constant uchar *, uint4, uint, float) {}
[[visible]] void postfixPrimary_f(
    thread uchar *state, constant uchar *, uint4 coordinate, uint, float value) {
    thread MPSKernelInOutPrefix *inout = reinterpret_cast<thread MPSKernelInOutPrefix *>(state);
    reinterpret_cast<device float *>(inout->buffers[0])[coordinate.x] = value;
}
[[visible]] void postfixPrimary_4xf(thread uchar *, constant uchar *, uint4, uint, float4) {}
[[visible]] void postfixPrimary_4xi(thread uchar *, constant uchar *, uint4, uint, int4) {}
[[visible]] void postfixPrimary_cf(
    thread uchar *state, constant uchar *, uint4 coordinate, uint, float real, float imaginary) {
    thread MPSKernelInOutPrefix *inout = reinterpret_cast<thread MPSKernelInOutPrefix *>(state);
    reinterpret_cast<device MPSComplexF32 *>(inout->buffers[0])[coordinate.x] = {real, imaginary};
}
[[visible]] void postfixPrimary_i(
    thread uchar *state, constant uchar *, uint4 coordinate, uint, int value) {
    thread MPSKernelInOutPrefix *inout = reinterpret_cast<thread MPSKernelInOutPrefix *>(state);
    reinterpret_cast<device int *>(inout->buffers[0])[coordinate.x] = value;
}
[[visible]] void postfixPrimary_i64(
    thread uchar *state, constant uchar *, uint4 coordinate, uint, long value) {
    thread MPSKernelInOutPrefix *inout = reinterpret_cast<thread MPSKernelInOutPrefix *>(state);
    reinterpret_cast<device long *>(inout->buffers[0])[coordinate.x] = value;
}
[[visible]] void postfixPrimary_u(
    thread uchar *state, constant uchar *, uint4 coordinate, uint, uint value) {
    thread MPSKernelInOutPrefix *inout = reinterpret_cast<thread MPSKernelInOutPrefix *>(state);
    reinterpret_cast<device uint *>(inout->buffers[0])[coordinate.x] = value;
}
[[visible]] void postfixPrimary_u16(thread uchar *, constant uchar *, uint4, uint, ushort) {}
[[visible]] void postfixPrimary_u64(
    thread uchar *state, constant uchar *, uint4 coordinate, uint, ulong value) {
    thread MPSKernelInOutPrefix *inout = reinterpret_cast<thread MPSKernelInOutPrefix *>(state);
    reinterpret_cast<device ulong *>(inout->buffers[0])[coordinate.x] = value;
}
[[visible]] void postfixPrimary_u8(thread uchar *, constant uchar *, uint4, uint, uchar) {}
[[visible]] void postfixPrimaryAccumulate_f(
    thread uchar *, constant uchar *, uint4, uint, float) {}
[[visible]] void postfixPrimaryAccumulate_4xf(
    thread uchar *, constant uchar *, uint4, uint, float4) {}
[[visible]] void postfixPrimaryAccumulate_cf(
    thread uchar *, constant uchar *, uint4, uint, float, float) {}
