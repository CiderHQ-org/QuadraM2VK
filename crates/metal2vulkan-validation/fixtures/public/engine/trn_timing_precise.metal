// Owned fixture: `trn_timing_fast`'s fast tangent kernel compiled without
// fast math (`-fno-fast-math`), so on MoltenVK, whose `Tan` is Metal's precise one in such an
// entry, `fast::tan` takes Apple's formula of the fast sine and cosine and the fast division's
// routine. Each thread applies the function to its running value 64 n times (64 steps written
// out in each of the loop's n iterations, as in `trn_timing_fast`), each step waiting for the
// last:
//
//   trn_time_fast_tan_precise   acc = fract(fast::tan(acc + y))
//
// Thread i reads n = in[0], its operand's bits y = in[1 + i % 64], and its start value's bits
// in[65 + i % 64], and writes the final value's bits to out[i]. Evidence scripts generate the
// cases (n, the operands, the thread count). The AIR's one loop is bounded by n.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

#define TRN_STEP4(s) s; s; s; s
#define TRN_STEP16(s) TRN_STEP4(s); TRN_STEP4(s); TRN_STEP4(s); TRN_STEP4(s)
#define TRN_STEP64(s) TRN_STEP16(s); TRN_STEP16(s); TRN_STEP16(s); TRN_STEP16(s)

kernel void trn_time_fast_tan_precise(device const uint *in [[buffer(0)]],
                                      device uint *out [[buffer(1)]],
                                      uint i [[thread_position_in_grid]])
{
    uint n = in[0];
    float y = as_type<float>(in[1 + (i & 63)]);
    float acc = as_type<float>(in[65 + (i & 63)]);
    for (uint k = 0; k < n; ++k) {
        TRN_STEP64(acc = fract(fast::tan(acc + y)));
    }
    out[i] = as_type<uint>(acc);
}
