// The format scout's functions (validation/src/bin/format-scout.rs), our own MSL: a
// fullscreen triangle, fragments writing a value of each class from the pixel's position, and
// kernels that read, sample, write and read-and-write a 16 x 16 texture of each class into a
// buffer of one value a texel.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

vertex Position scout_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    Position out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fragment float4 scout_fragment_f(Position in [[stage_in]]) {
    float2 p = floor(in.position.xy);
    return float4(fract(p.x / 7.0), fract(p.y / 5.0), 0.25, 0.75);
}

fragment int4 scout_fragment_i(Position in [[stage_in]]) {
    int2 p = int2(in.position.xy);
    return int4(p.x - 8, p.y - 8, -3, 5);
}

fragment uint4 scout_fragment_u(Position in [[stage_in]]) {
    uint2 p = uint2(in.position.xy);
    return uint4(p.x, p.y, 3, 5);
}

kernel void scout_read_f(texture2d<float, access::read> t [[texture(0)]],
                         device float4 *out [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 16 + g.x] = t.read(g);
}

kernel void scout_read_i(texture2d<int, access::read> t [[texture(0)]],
                         device int4 *out [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 16 + g.x] = t.read(g);
}

kernel void scout_read_u(texture2d<uint, access::read> t [[texture(0)]],
                         device uint4 *out [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 16 + g.x] = t.read(g);
}

kernel void scout_read_d(depth2d<float, access::read> t [[texture(0)]],
                         device float *out [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 16 + g.x] = t.read(g);
}

kernel void scout_sample_f(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                           device float4 *out [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    float2 uv = (float2(g) + float2(0.25, 0.75)) / 16.0;
    out[g.y * 16 + g.x] = t.sample(s, uv);
}

kernel void scout_sample_d(depth2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                           device float *out [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    float2 uv = (float2(g) + float2(0.25, 0.75)) / 16.0;
    out[g.y * 16 + g.x] = t.sample(s, uv);
}

kernel void scout_write_f(texture2d<float, access::write> t [[texture(0)]],
                          uint2 g [[thread_position_in_grid]]) {
    t.write(float4(fract(g.x / 7.0), fract(g.y / 5.0), 0.25, 0.75), g);
}

kernel void scout_write_i(texture2d<int, access::write> t [[texture(0)]],
                          uint2 g [[thread_position_in_grid]]) {
    t.write(int4(int(g.x) - 8, int(g.y) - 8, -3, 5), g);
}

kernel void scout_write_u(texture2d<uint, access::write> t [[texture(0)]],
                          uint2 g [[thread_position_in_grid]]) {
    t.write(uint4(g.x, g.y, 3, 5), g);
}

kernel void scout_read_write_f(texture2d<float, access::read_write> t [[texture(0)]],
                               uint2 g [[thread_position_in_grid]]) {
    t.write(t.read(g).yxwz, g);
}

kernel void scout_read_write_i(texture2d<int, access::read_write> t [[texture(0)]],
                               uint2 g [[thread_position_in_grid]]) {
    t.write(t.read(g).yxwz, g);
}

kernel void scout_read_write_u(texture2d<uint, access::read_write> t [[texture(0)]],
                               uint2 g [[thread_position_in_grid]]) {
    t.write(t.read(g).yxwz, g);
}
