// regroup_chain: Metal's pairing of a fast-math multiply chain's lane-uniform factors
// (`lower::arith::regroup`). Each entry computes a vector product `n = s * x` and multiplies
// lanes of it by `a`; Metal multiplies the two lane-uniform factors first where they repeat in
// every lane's chain.
// - dither: a fragment shape: a noise texel less one half, times a uniform scale, its first three
//   lanes times a sampled lane plus the sample, its fourth lane stored as alpha. Metal computes
//   `x * (scale * a)` fused into the addition.
// - pair_lanes: the kernel form, the product alone, the fourth lane stored too.
// - lanes_used_twice: the whole product is stored as well: its lanes are leaves, and the chain
//   keeps its grouping (and notes the rewrites Metal may make).
// - per_lane_third: the outer factor differs per lane: Metal orders the factors by their rank, so
//   the chain keeps its grouping and notes the rewrites.
// - already_paired: `x * (s * a)` as written: Metal keeps it.
#include <metal_stdlib>
using namespace metal;

struct DitherIn {
    float4 position [[position]];
    float2 uv;
};

fragment half4 dither(DitherIn in [[stage_in]], texture2d<half> image [[texture(0)]],
                      texture2d<half> noise [[texture(1)]], sampler s [[sampler(0)]],
                      constant half &scale [[buffer(0)]]) {
    half4 h = image.sample(s, in.uv);
    uint2 q = uint2(in.position.xy);
    half4 n = (noise.read(q % 32u) - 0.5h) * scale;
    half3 d = n.xyz * h.y + h.xyz;
    return half4(d, n.w);
}

kernel void pair_lanes(device const float4 *X [[buffer(0)]], constant float &S [[buffer(1)]],
                       device const float4 *A [[buffer(2)]], device float4 *R [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    float4 x = X[gid];
    float s = S;
    float a = A[gid].y;
    float4 n = s * x;
    float3 m = n.xyz * a;
    R[gid] = float4(m, n.w);
}

kernel void lanes_used_twice(device const half4 *X [[buffer(0)]], constant half &S [[buffer(1)]],
                             device const half4 *A [[buffer(2)]], device half4 *R [[buffer(3)]],
                             device half4 *R2 [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    half4 x = X[gid];
    half s = S;
    half a = A[gid].x;
    half4 n = s * x;
    half3 m = n.xyz * a + X[gid + 1].xyz;
    R[gid] = half4(m, 1.0h);
    R2[gid] = n;
}

kernel void per_lane_third(device const half4 *X [[buffer(0)]], constant half &S [[buffer(1)]],
                           device const half4 *A [[buffer(2)]], device half4 *R [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    half4 x = X[gid];
    half s = S;
    half3 a = A[gid].xyz;
    half4 n = s * x;
    half3 m = n.xyz * a + X[gid + 1].xyz;
    R[gid] = half4(m, n.w);
}

kernel void already_paired(device const half4 *X [[buffer(0)]], constant half &S [[buffer(1)]],
                           device const half4 *A [[buffer(2)]], device half4 *R [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    half4 x = X[gid];
    half s = S;
    half a = A[gid].y;
    half sa = s * a;
    half3 m = x.xyz * sa + X[gid + 1].xyz;
    R[gid] = half4(m, 1.0h);
}
