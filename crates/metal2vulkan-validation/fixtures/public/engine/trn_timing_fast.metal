// Owned fixture: timing kernels for the math functions computed with
// the translation's routines, compiled with the default fast math, so the precise calls are
// spelled `precise::` (on MoltenVK a precise call in a fast-math entry takes the routine). Each
// thread applies its function to its running value 64 n times (64 steps written out in each of
// the loop's n iterations), each step waiting for the last:
//
//   trn_time_fast_sin   acc = fast::sin(acc + y)
//   trn_time_fast_tan   acc = fract(fast::tan(acc + y))
//   trn_time_sin        acc = precise::sin(acc + y)
//   trn_time_atan       acc = precise::atan(acc + y)
//   trn_time_atan2      acc = precise::atan2(acc + y, z)
//   trn_time_sqrt       acc = precise::sqrt(acc + y)
//   trn_time_rsqrt      acc = precise::rsqrt(acc + y)
//   trn_time_fmod       acc = precise::fmod(acc + y, z)
//   trn_time_sinpi      acc = sinpi(acc + y)
//
// Thread i reads n = in[0], its operand's bits y = in[1 + i % 64], its start value's bits
// in[65 + i % 64], and its second operand's bits z = in[129 + i % 64], and writes the final
// value's bits to out[i]. Evidence scripts generate the cases (n, the operands, the thread
// count): the operands decide which arm of a routine runs (inside or past pi for the fast sine
// and tangent, the exponent gap of fmod). The AIR's one loop per kernel is bounded by n. The
// steps are written out, not looped, so the Metal oracle's loop fuel, which a dispatch's
// iterations of all its threads draw on together, bounds n alone: 2^18 threads may take n up to
// 2^11 under it while each runs 2^17 steps.
#include <metal_stdlib>
using namespace metal;

#define TRN_STEP4(s) s; s; s; s
#define TRN_STEP16(s) TRN_STEP4(s); TRN_STEP4(s); TRN_STEP4(s); TRN_STEP4(s)
#define TRN_STEP64(s) TRN_STEP16(s); TRN_STEP16(s); TRN_STEP16(s); TRN_STEP16(s)

#define TRN_TIME(name, step)                                                        \
    kernel void name(device const uint *in [[buffer(0)]],                           \
                     device uint *out [[buffer(1)]],                                \
                     uint i [[thread_position_in_grid]])                            \
    {                                                                               \
        uint n = in[0];                                                             \
        float y = as_type<float>(in[1 + (i & 63)]);                                 \
        float acc = as_type<float>(in[65 + (i & 63)]);                              \
        float z = as_type<float>(in[129 + (i & 63)]);                               \
        (void)z;                                                                    \
        for (uint k = 0; k < n; ++k) {                                              \
            TRN_STEP64(acc = step);                                                 \
        }                                                                           \
        out[i] = as_type<uint>(acc);                                                \
    }

TRN_TIME(trn_time_fast_sin, fast::sin(acc + y))
TRN_TIME(trn_time_fast_tan, fract(fast::tan(acc + y)))
TRN_TIME(trn_time_sin, precise::sin(acc + y))
TRN_TIME(trn_time_atan, precise::atan(acc + y))
TRN_TIME(trn_time_atan2, precise::atan2(acc + y, z))
TRN_TIME(trn_time_sqrt, precise::sqrt(acc + y))
TRN_TIME(trn_time_rsqrt, precise::rsqrt(acc + y))
TRN_TIME(trn_time_fmod, precise::fmod(acc + y, z))
TRN_TIME(trn_time_sinpi, sinpi(acc + y))
