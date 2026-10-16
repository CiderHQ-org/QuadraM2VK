// Owned fixture: Metal's `pow` at every special operand of C99 Annex F (F.9.4.4), without fast math
// (`-fno-fast-math`: no call carries `nnan` or `ninf`, so a NaN or infinite operand asks a defined
// question, and the symbols are the precise ones). The translator lowers `pow` as C99 around
// `GLSL.std.450 Pow(|x|, y)`: `pow(x, +-0) = 1` and `pow(+1, y) = 1` even for a NaN, `pow(-1,
// +-inf) = 1`, signed zeros and infinities by the exponent's parity, NaN for a negative finite base
// to a non-integral exponent, and NaN for any other NaN operand; the Metal side of each of those
// rules is unmeasured. The public `kernel_pow_signed_zero_and_infinite_base` asks a 7 x 12 grid
// under fast math, where a NaN or infinite operand is poison.
//
// Lane i reads `in[i]` = (float base bits, float exponent bits, half base bits | half exponent
// bits << 16, 0) and writes three words: `precise::pow` (`air.pow.f32`), `fast::pow`
// (`air.fast_pow.f32`), and the half `pow` (`air.pow.f16`) zero-extended. The lanes are the
// grid of 20 bases (+-0, +-1, +-0.5, +-2, +-inf, +-NaN, +-(1 - ulp), +-(1 + ulp), +-smallest
// denormal, +-largest finite) and 26 exponents (+-0, +-1, +-2, +-3, +-0.5, +-2.5, +-inf, NaN,
// +-the largest odd integer, the smallest even integer past it, +-a large even integer,
// +-smallest denormal, the top power of two, one past it, the bottom normal power, and an odd
// power below the denormals), each with its half analogue, restricted to the pairs whose C99
// answer is a special value, a signed zero or infinity by underflow or overflow, or an exactly
// representable normal in both widths (432 lanes). Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void a1_pow_special_values(
    device const uint4 *in [[buffer(0)]],
    device uint *out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    uint4 v = in[i];
    float b = as_type<float>(v.x);
    float e = as_type<float>(v.y);
    half hb = as_type<half>(ushort(v.z));
    half he = as_type<half>(ushort(v.z >> 16));
    out[3 * i + 0] = as_type<uint>(precise::pow(b, e));
    out[3 * i + 1] = as_type<uint>(fast::pow(b, e));
    out[3 * i + 2] = uint(as_type<ushort>(pow(hb, he)));
}
