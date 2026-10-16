// Owned fixture: how Metal rounds a fused multiply-add and a written
// `a * b + c`, compiled with the default fast math. `fma` is `air.fma.f32` / `air.fma.f16`, which
// MSL Tables 8.1 and 8.2 make correctly rounded; `a * b + c` is an `fmul fast` feeding an
// `fadd fast` (both carry `contract`), which Metal's compiler may fuse or not. `fmx_fma_precise`
// is the same program without fast math, where `a * b + c` is `llvm.fmuladd`.
//
// Thread i reads `fin[i]` = float patterns (a, b, c, w) and `hin[i]` = half patterns
// (a, b, c, 0), and writes four words: `fma(a, b, c)`, `a * b + c`, the half
// `fma(a, b, c) | (a * b + c) << 16`, and `a * w` (a lone product: whether Metal flushes a
// denormal product). The float lanes are where one rounding and two differ, where computing in
// `f64` and rounding to `float` would round twice, signed zeros, infinities, NaNs, a product past
// the largest float that the addend brings back, and denormal operands, products, and sums. The
// half lanes are where one rounding differs from a rounding to `float` first, and from two
// roundings to `half`. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void fmx_fma_fast(device const uint4 *fin [[buffer(0)]],
                         device const ushort4 *hin [[buffer(1)]],
                         device uint4 *out [[buffer(2)]],
                         uint i [[thread_position_in_grid]])
{
    float4 f = as_type<float4>(fin[i]);
    half4 h = as_type<half4>(hin[i]);
    half hf = fma(h.x, h.y, h.z);
    half he = h.x * h.y + h.z;
    out[i] = uint4(as_type<uint>(fma(f.x, f.y, f.z)), as_type<uint>(f.x * f.y + f.z),
                   uint(as_type<ushort>(hf)) | uint(as_type<ushort>(he)) << 16,
                   as_type<uint>(f.x * f.w));
}
