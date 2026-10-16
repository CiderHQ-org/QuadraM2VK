// Owned MSL: the read sampler used by a texture read in another
// block than its definition, a dynamic gather offset (a signed `Offset` operand), and a bicubic
// constexpr sampler (refused: Metal's bicubic weights are not established). Every function is
// loop-free.
#include <metal_stdlib>
using namespace metal;

constexpr sampler bicubic_clamp(filter::bicubic, address::clamp_to_edge);

// The front end calls `air.get_read_sampler` once in the entry block and passes it to both reads,
// the second in the branch.
kernel void read_across_blocks(texture2d<float> t [[texture(0)]],
                               device float4 *out [[buffer(0)]],
                               constant uint &n [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
    float4 acc = t.read(gid);
    if (gid.x < n) {
        acc += t.read(gid + uint2(1, 0));
    }
    out[gid.y * 4 + gid.x] = acc;
}

// The same shape in a fragment function.
fragment float4 fragment_read_across_blocks(float4 pos [[position]],
                                            texture2d<float> t [[texture(0)]],
                                            constant uint &n [[buffer(0)]]) {
    uint2 p = uint2(pos.xy);
    float4 acc = t.read(p);
    if (p.x < n) {
        acc += t.read(p + uint2(1, 0));
    }
    return acc;
}

// A dynamic gather offset: the clamped `Offset` operand, a signed vector as MSL's `gather` takes
//.
kernel void gather_dynamic_offset(texture2d<float> t [[texture(0)]],
                                  sampler s [[sampler(0)]],
                                  device float4 *out [[buffer(0)]],
                                  uint gid [[thread_position_in_grid]]) {
    int2 o = int2(int(gid & 7) - 4, 1);
    out[gid] = t.gather(s, float2(gid) * 0.25, o);
}

// A bicubic constexpr sampler: `NotImplemented(SamplerBicubicFilter)` on every profile.
kernel void sample_bicubic(texture2d<float> t [[texture(0)]],
                           device float4 *out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(bicubic_clamp, float2(gid) * 0.25, level(0.0));
}
