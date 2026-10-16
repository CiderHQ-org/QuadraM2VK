// Owned MSL: memory intrinsics of a runtime length. Thread 0 copies
// `lengths.copy` bytes (a `ulong`: an `i64` length, held as two 32-bit words on a device without
// `shaderInt64`) from `src` to `dst`, both word-aligned, so the copy is whole words
// then tail bytes; then it fills `lengths.fill` bytes (a `uint` count widened to `i64`) at byte
// 512 of `dst` with 0x5a. Every other thread returns. The only loops are the lowered copy and
// fill, which the execution bound instruments.
#include <metal_stdlib>
using namespace metal;

struct Lengths {
    ulong copy;
    uint fill;
};

kernel void kernel_copy_runtime_ulong_length(device const uint *src [[buffer(0)]],
                                             device uint *dst [[buffer(1)]],
                                             constant Lengths &lengths [[buffer(2)]],
                                             uint tid [[thread_position_in_grid]]) {
    if (tid != 0) {
        return;
    }
    __builtin_memcpy(dst, src, lengths.copy);
    __builtin_memset((device uchar *)dst + 512, 0x5a, lengths.fill);
}
