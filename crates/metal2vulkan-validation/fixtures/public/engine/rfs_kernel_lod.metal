// Owned fixture: `calculate_clamped_lod` and `calculate_unclamped_lod` outside a
// fragment function, which MSL calls undefined and Metal answers deterministically (measured on the
// M3 Max and an M4): the unclamped level is 0, and the clamped one is the sampler's minimum level
// bounded by the texture's last level. Runtime samplers over 2D, 2D-array, 3D and cube textures;
// constexpr samplers over 2D, 3D and cube textures with LOD clamps on and off Metal's grid of
// sixty-fourths, under every mip filter; a vertex function's query; and a depth texture's, which
// the translation refuses (not measured).
//
// Thread i of an 8x8 grid queries at the coordinate (gid * p.xy + p.zw) / 64 (a cube: the
// direction (1, that * 2 - 1)); out[2 i] is the clamped level and out[2 i + 1] the unclamped one.
#include <metal_stdlib>
using namespace metal;

#define LOD_BODY(T, S, C)                                                                         \
  uint i = gid.y * 8 + gid.x;                                                                     \
  out[2 * i] = T.calculate_clamped_lod(S, C);                                                     \
  out[2 * i + 1] = T.calculate_unclamped_lod(S, C);

#define UV ((float2(gid) * p.xy + p.zw) / 64.0f)
#define UVW (float3(float2(gid) * p.xy + p.zw, float(gid.x) * p.x + p.z) / 64.0f)
#define DIR (float3(1.0f, (float2(gid) * p.xy + p.zw) / 32.0f - 1.0f))

kernel void lod_2d(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                   device float *out [[buffer(0)]], constant float4 &p [[buffer(1)]],
                   uint2 gid [[thread_position_in_grid]]) {
  LOD_BODY(t, s, UV)
}

kernel void lod_2d_array(texture2d_array<float> t [[texture(0)]], sampler s [[sampler(0)]],
                         device float *out [[buffer(0)]], constant float4 &p [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
  LOD_BODY(t, s, UV)
}

kernel void lod_3d(texture3d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                   device float *out [[buffer(0)]], constant float4 &p [[buffer(1)]],
                   uint2 gid [[thread_position_in_grid]]) {
  LOD_BODY(t, s, UVW)
}

kernel void lod_cube(texturecube<float> t [[texture(0)]], sampler s [[sampler(0)]],
                     device float *out [[buffer(0)]], constant float4 &p [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
  LOD_BODY(t, s, DIR)
}

// Constexpr samplers: a linear mip filter clamped off the grid (0.3 reads as the half 0.30005,
// 19/64 on the grid), a nearest one (2.7: level 3), no mip filter (always 0), a linear one without
// a clamp (0), and a linear one clamped past the last level (10: the last level).
kernel void lod_2d_linear_clamp(texture2d<float> t [[texture(0)]], device float *out [[buffer(0)]],
                                constant float4 &p [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
  constexpr sampler s(filter::linear, mip_filter::linear, lod_clamp(0.3f, 4.0f));
  LOD_BODY(t, s, UV)
}

kernel void lod_2d_nearest_clamp(texture2d<float> t [[texture(0)]], device float *out [[buffer(0)]],
                                 constant float4 &p [[buffer(1)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  constexpr sampler s(filter::linear, mip_filter::nearest, lod_clamp(2.7f, 4.0f));
  LOD_BODY(t, s, UV)
}

kernel void lod_2d_unmipped_clamp(texture2d<float> t [[texture(0)]], device float *out [[buffer(0)]],
                                  constant float4 &p [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  constexpr sampler s(filter::linear, lod_clamp(2.5f, 4.0f));
  LOD_BODY(t, s, UV)
}

kernel void lod_2d_linear_open(texture2d<float> t [[texture(0)]], device float *out [[buffer(0)]],
                               constant float4 &p [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
  constexpr sampler s(filter::linear, mip_filter::linear);
  LOD_BODY(t, s, UV)
}

kernel void lod_2d_linear_past(texture2d<float> t [[texture(0)]], device float *out [[buffer(0)]],
                               constant float4 &p [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
  constexpr sampler s(filter::nearest, mip_filter::linear, lod_clamp(10.0f, 20.0f));
  LOD_BODY(t, s, UV)
}

// A nearest mip filter's tie: 1.5 reads level 1 (the fraction must exceed one half).
kernel void lod_3d_nearest_tie(texture3d<float> t [[texture(0)]], device float *out [[buffer(0)]],
                               constant float4 &p [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
  constexpr sampler s(filter::nearest, mip_filter::nearest, lod_clamp(1.5f, 4.0f));
  LOD_BODY(t, s, UVW)
}

// 1.1 reads as the half 1.0996, 70/64 on the grid; the maximum clamp does not move it.
kernel void lod_cube_linear_clamp(texturecube<float> t [[texture(0)]], device float *out [[buffer(0)]],
                                  constant float4 &p [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  constexpr sampler s(filter::linear, mip_filter::linear, lod_clamp(1.1f, 1.9f));
  LOD_BODY(t, s, DIR)
}

// A vertex function's query, vertex v standing for thread (v % 8, v / 8).
vertex void lod_vertex(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                       device float *out [[buffer(0)]], constant float4 &p [[buffer(1)]],
                       uint vid [[vertex_id]]) {
  uint2 gid = uint2(vid % 8, vid / 8);
  LOD_BODY(t, s, UV)
}

// A depth texture's query: not measured, so refused.
kernel void lod_depth(depth2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                      device float *out [[buffer(0)]], constant float4 &p [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
  LOD_BODY(t, s, UV)
}
