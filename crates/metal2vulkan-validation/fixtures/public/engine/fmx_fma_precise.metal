// Owned fixture: `fmx_fma_fast` without fast math (`-fno-fast-math`). The
// same program: `fma` is still `air.fma.f32` / `air.fma.f16`, and `a * b + c` is now
// `llvm.fmuladd.f32` / `llvm.fmuladd.f16` (the front end contracts the written expression into
// one call that the implementation may fuse or not), so the pair reads whether Metal fuses
// `llvm.fmuladd`, and whether it rounds a `half` fused multiply-add once.
// metal-fast-math: off
//
// Thread i reads `fin[i]` = float patterns (a, b, c, w) and `hin[i]` = half patterns
// (a, b, c, 0), and writes four words: `fma(a, b, c)`, `a * b + c`, the half
// `fma(a, b, c) | (a * b + c) << 16`, and `a * w`. The lanes are `fmx_fma_fast`'s. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void fmx_fma_precise(device const uint4 *fin [[buffer(0)]],
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
