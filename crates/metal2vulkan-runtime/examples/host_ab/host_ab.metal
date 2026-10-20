// The host A/B benchmark's shaders (examples/host_ab): one source, compiled once into a metallib
// that both sides load, natively and through the runtime's translation.
//
//   xcrun -sdk macosx metal -DNONCE=<n> -o host_ab.metallib host_ab.metal
//
// NONCE (below 2^24, so each value is its own float) only changes a constant in each function,
// so each value gives functions of the same shape whose bytes differ: a cold pipeline creation
// finds them in no cache, Metal's or the runtime's.

#include <metal_stdlib>
using namespace metal;

#ifndef NONCE
#define NONCE 0
#endif

struct QuadOut {
    float4 position [[position]];
    float2 uv;
};

// A rectangle (x0, y0, x1, y1 in normalized device coordinates) as a strip of four vertices.
vertex QuadOut quad_vertex(uint vid [[vertex_id]], constant float4 &rect [[buffer(0)]]) {
    float2 corner = float2(float(vid & 1u), float(vid >> 1));
    QuadOut out;
    out.position = float4(mix(rect.xy, rect.zw, corner), float(NONCE) * 1.0e-12, 1.0);
    out.uv = float2(corner.x, 1.0 - corner.y);
    return out;
}

// The small draws' fragment: one sample.
fragment float4 textured_fragment(QuadOut in [[stage_in]],
                                  texture2d<float> tex [[texture(0)]],
                                  sampler smp [[sampler(0)]]) {
    return tex.sample(smp, in.uv) + float4(float(NONCE) * 1.0e-12);
}

// The heavy work's parameters, read from a constant buffer.
struct HeavyParams {
    float4 coeffs[64];
    uint iterations;
    float scale;
};

// The heavy fragment: float ALU, half math, texture sampling and constant-buffer reads in a loop.
fragment half4 heavy_fragment(QuadOut in [[stage_in]],
                              constant HeavyParams &p [[buffer(0)]],
                              texture2d<half> tex [[texture(0)]],
                              sampler smp [[sampler(0)]]) {
    float2 uv = in.uv;
    float acc = float(NONCE) * 1.0e-12;
    half4 hacc = half4(0.0h);
    for (uint i = 0; i < p.iterations; i++) {
        float4 c = p.coeffs[i & 63u];
        acc = fma(acc, c.x, sin(uv.x * c.y + acc) * c.z);
        half4 t = tex.sample(smp, uv + float2(c.w, acc) * 0.01);
        hacc = fma(hacc, half4(half(c.x)), t * 0.5h);
        uv = fract(uv * 1.37 + c.zw);
    }
    return hacc + half4(half(acc * p.scale));
}

// The heavy kernel: float ALU, half math, texture reads, constant-buffer reads and threadgroup
// memory with barriers, in a loop. Threadgroups of 256 threads.
kernel void heavy_kernel(device float4 *out [[buffer(0)]],
                         constant HeavyParams &p [[buffer(1)]],
                         texture2d<half> tex [[texture(0)]],
                         uint tid [[thread_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float shared_f[256];
    threadgroup half shared_h[256];
    float acc = float(tid) * 1.0e-6 + float(NONCE) * 1.0e-12;
    half hacc = 0.0h;
    for (uint i = 0; i < p.iterations; i++) {
        float4 c = p.coeffs[(i + lid) & 63u];
        acc = fma(acc, c.x, sin(acc * c.y) * c.z + c.w);
        half4 t = tex.read(uint2((tid + i * 7u) & 255u, ((tid >> 8) + i) & 255u));
        hacc = fma(hacc, half(c.x), dot(t, half4(0.25h)));
        shared_f[lid] = acc;
        shared_h[lid] = hacc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += shared_f[(lid + 1u) & 255u] * 0.5;
        hacc += shared_h[(lid + 7u) & 255u] * 0.5h;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = float4(acc * p.scale, float(hacc), 0.0, 1.0);
}

// The small dispatches' kernels: two, so a dispatch can change the pipeline state.
kernel void tiny_kernel(device uint *out [[buffer(0)]],
                        constant uint &value [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {
    out[tid] = value + tid + uint(NONCE);
}

kernel void tiny_kernel_b(device uint *out [[buffer(0)]],
                          constant uint &value [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    out[tid] = value * 3u + tid + uint(NONCE);
}
