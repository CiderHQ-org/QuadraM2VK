// Samples whose level the device's level arithmetic takes from derivatives or gradients
// (`PrecisionNote::DeviceSampleLevel`), beside samples whose level chooses nothing or another note
// records. Every function is loop-free.
#include <metal_stdlib>
using namespace metal;

constexpr sampler nearest_mips(filter::nearest, mip_filter::nearest);
constexpr sampler nearest_base(filter::nearest);
constexpr sampler linear_mips(filter::linear, mip_filter::linear);

// Nearest filters under a nearest mip filter: the device's level chooses the texel.
fragment float4 dl_nearest_mips(float4 pos [[position]], texture2d<float> t [[texture(0)]]) {
    return t.sample(nearest_mips, pos.xy * 0.0625);
}

// The same with a bias within every device's limit, which the device's level arithmetic applies.
fragment float4 dl_nearest_mips_bias(float4 pos [[position]], texture2d<float> t [[texture(0)]]) {
    return t.sample(nearest_mips, pos.xy * 0.0625, bias(0.75));
}

// Nearest filters without a mip filter read the base level: the level chooses nothing.
fragment float4 dl_nearest_base(float4 pos [[position]], texture2d<float> t [[texture(0)]]) {
    return t.sample(nearest_base, pos.xy * 0.0625);
}

// A filtered sample, which its filter note records.
fragment float4 dl_linear_mips(float4 pos [[position]], texture2d<float> t [[texture(0)]]) {
    return t.sample(linear_mips, pos.xy * 0.0625);
}

// An integer texture under a nearest mip filter.
fragment uint4 dl_uint_mips(float4 pos [[position]], texture2d<uint> t [[texture(0)]]) {
    return t.sample(nearest_mips, pos.xy * 0.0625);
}

// A runtime sampler, whose word states its filters.
fragment float4 dl_runtime(float4 pos [[position]], texture2d<float> t [[texture(0)]],
                           sampler s [[sampler(0)]]) {
    return t.sample(s, pos.xy * 0.0625);
}

// An LOD query, which answers the level.
fragment float4 dl_query(float4 pos [[position]], texture2d<float> t [[texture(0)]]) {
    return float4(t.calculate_unclamped_lod(nearest_base, pos.xy * 0.0625));
}

// A cube's implicit level, which `CubeGradientFootprint` records.
fragment float4 dl_cube(float4 pos [[position]], texturecube<float> t [[texture(0)]]) {
    return t.sample(nearest_mips, float3(pos.xy * 0.0625, 1.0));
}

// Explicit gradients under a nearest mip filter.
kernel void dl_gradients(texture2d<float> t [[texture(0)]], device float4 *out [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(nearest_mips, float2(gid) * 0.125,
                        gradient2d(float2(0.5, 0.0), float2(0.0, 0.5)));
}

// An explicit level, which the module hands the device on Metal's grid.
kernel void dl_level(texture2d<float> t [[texture(0)]], device float4 *out [[buffer(0)]],
                     uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(nearest_mips, float2(gid) * 0.125, level(1.25));
}

// An integer texture's explicit level, which Metal's rule does not reach: the device rounds it.
kernel void dl_uint_level(texture2d<uint> t [[texture(0)]], device uint4 *out [[buffer(0)]],
                          uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(nearest_mips, float2(gid) * 0.125, level(1.5));
}

// An integer texture's whole level, which the device reads as Metal does.
kernel void dl_uint_whole_level(texture2d<uint> t [[texture(0)]], device uint4 *out [[buffer(0)]],
                                uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(nearest_mips, float2(gid) * 0.125, level(2.0));
}
