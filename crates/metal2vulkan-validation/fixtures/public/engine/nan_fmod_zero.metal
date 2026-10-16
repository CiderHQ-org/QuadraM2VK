// Owned fixture (NaN model): the sign of Metal's zero remainder, and its remainder
// by an infinity, for `float` and `half` `fmod`. Lane `i` reads floats a and b (`in[i].xy`) and
// halves ha and hb (the low and high halves of `in[i].z`) and writes four words to `out[i]`:
// fmod(a, b), fmod(ha, hb) in the low half, a, and b. `nan_fmod_zero_fast` is the same program
// under the default fast math. Compiled without fast math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_fmod_zero(device const uint4 *in [[buffer(0)]],
                          device uint4 *out [[buffer(1)]],
                          uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y);
    half ha = as_type<half>(ushort(in[tid].z)), hb = as_type<half>(ushort(in[tid].z >> 16));
    out[tid] = uint4(as_type<uint>(fmod(a, b)), uint(as_type<ushort>(fmod(ha, hb))),
                     as_type<uint>(a), as_type<uint>(b));
}
