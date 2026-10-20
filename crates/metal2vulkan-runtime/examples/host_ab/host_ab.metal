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

// The feature kernels: each one Metal feature of the heavy kernel in the same loop, so their
// ratios say which feature its cost comes from. Each takes the heavy kernel's bindings
// (`out`, `p`, `tex`) and `src`, a device buffer as large as `out`.
kernel void k_fma(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    float a = float(tid) * 1.0e-6 + float(NONCE) * 1.0e-12;
    float b = 1.0;
    for (uint i = 0; i < p.iterations; i++) {
        a = fma(a, 0.999, b);
        b = fma(b, 1.0001, -a * 0.5);
    }
    out[tid] = float4(a, b, 0.0, 1.0);
}

kernel void k_sin(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    float a = float(tid) * 1.0e-6 + float(NONCE) * 1.0e-12;
    for (uint i = 0; i < p.iterations; i++) {
        a = sin(a * 1.3 + 0.7);
    }
    out[tid] = float4(a, 0.0, 0.0, 1.0);
}

kernel void k_half(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                   uint tid [[thread_position_in_grid]]) {
    half a = half(tid & 255u) * 0.01h + half(float(NONCE) * 1.0e-12);
    half b = 0.5h;
    for (uint i = 0; i < p.iterations; i++) {
        a = fma(a, 0.999h, b);
        b = b * 0.75h + a * 0.125h;
    }
    out[tid] = float4(float(a), float(b), 0.0, 1.0);
}

kernel void k_constant(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
    float4 acc = float4(float(NONCE) * 1.0e-12);
    for (uint i = 0; i < p.iterations; i++) {
        acc = fma(acc, float4(0.5), p.coeffs[(i + tid) & 63u]);
    }
    out[tid] = acc;
}

kernel void k_device(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                     device const float4 *src [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    float4 acc = float4(float(NONCE) * 1.0e-12);
    for (uint i = 0; i < p.iterations; i++) {
        acc = fma(acc, float4(0.5), src[(tid + i * 4099u) & 1048575u]);
    }
    out[tid] = acc;
}

kernel void k_texture(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                      texture2d<half> tex [[texture(0)]],
                      uint tid [[thread_position_in_grid]]) {
    half4 acc = half4(half(float(NONCE) * 1.0e-12));
    for (uint i = 0; i < p.iterations; i++) {
        acc = fma(acc, half4(0.5h), tex.read(uint2((tid + i * 7u) & 255u, ((tid >> 8) + i) & 255u)));
    }
    out[tid] = float4(acc);
}

kernel void k_texture_float(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                            texture2d<float> tex [[texture(0)]],
                            uint tid [[thread_position_in_grid]]) {
    float4 acc = float4(float(NONCE) * 1.0e-12);
    for (uint i = 0; i < p.iterations; i++) {
        acc = fma(acc, float4(0.5), tex.read(uint2((tid + i * 7u) & 255u, ((tid >> 8) + i) & 255u)));
    }
    out[tid] = acc;
}

kernel void k_threadgroup_float(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                                uint tid [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float shared_f[256];
    float acc = float(tid) * 1.0e-6 + float(NONCE) * 1.0e-12;
    for (uint i = 0; i < p.iterations; i++) {
        shared_f[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = fma(shared_f[(lid + 1u) & 255u], 0.5, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = float4(acc, 0.0, 0.0, 1.0);
}

kernel void k_threadgroup_half(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                               uint tid [[thread_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]]) {
    threadgroup half shared_h[256];
    half acc = half(tid & 255u) * 0.01h + half(float(NONCE) * 1.0e-12);
    for (uint i = 0; i < p.iterations; i++) {
        shared_h[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = fma(shared_h[(lid + 7u) & 255u], 0.5h, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = float4(float(acc), 0.0, 0.0, 1.0);
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

// The many-draws workload's functions: a small rectangle per draw, of its own uniforms, and four
// fragment functions over two textures and one sampler.
struct DrawUniforms {
    float4 rect;   // x, y, width, height in NDC
    float4 color;
    float4 misc;   // x: depth, y: phase
};

struct DrawOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex DrawOut draws_vertex(uint vid [[vertex_id]],
                            const device float2 *quad [[buffer(0)]],
                            constant DrawUniforms &u [[buffer(1)]],
                            constant float4 &jitter [[buffer(2)]]) {
    float2 p = quad[vid];
    DrawOut out;
    out.position = float4(u.rect.xy + p * u.rect.zw + jitter.xy, u.misc.x, 1.0);
    out.uv = p;
    out.color = u.color + float4(float(NONCE) * 1.0e-12);
    return out;
}

fragment float4 draws_fragment0(DrawOut in [[stage_in]],
                                texture2d<float> t0 [[texture(0)]],
                                texture2d<float> t1 [[texture(1)]],
                                sampler s [[sampler(0)]],
                                constant DrawUniforms &u [[buffer(0)]]) {
    return t0.sample(s, in.uv) * in.color;
}

fragment float4 draws_fragment1(DrawOut in [[stage_in]],
                                texture2d<float> t0 [[texture(0)]],
                                texture2d<float> t1 [[texture(1)]],
                                sampler s [[sampler(0)]],
                                constant DrawUniforms &u [[buffer(0)]]) {
    return mix(t0.sample(s, in.uv), t1.sample(s, in.uv * 2.0), 0.5) * in.color;
}

fragment float4 draws_fragment2(DrawOut in [[stage_in]],
                                texture2d<float> t0 [[texture(0)]],
                                texture2d<float> t1 [[texture(1)]],
                                sampler s [[sampler(0)]],
                                constant DrawUniforms &u [[buffer(0)]]) {
    float r = length(in.uv - 0.5);
    return float4(in.color.rgb * (1.0 - r), 0.8) * t1.sample(s, in.uv);
}

fragment float4 draws_fragment3(DrawOut in [[stage_in]],
                                texture2d<float> t0 [[texture(0)]],
                                texture2d<float> t1 [[texture(1)]],
                                sampler s [[sampler(0)]],
                                constant DrawUniforms &u [[buffer(0)]]) {
    return float4(in.uv, fract(u.misc.y), 1.0) * t0.sample(s, in.uv.yx);
}
