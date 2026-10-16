// Owned fixture: the texture and sampler arguments whose
// reflection Metal answers (MTLTextureBinding): every texture type a kernel takes, each access,
// depth textures, a texture array argument, and an unused texture. Written for this repository;
// no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void reflection_textures(texture2d<float, access::read> r [[texture(0)]],
                                texture2d<half, access::write> w [[texture(1)]],
                                texture2d<uint, access::read_write> rw [[texture(2)]],
                                texture2d<float> s [[texture(3)]],
                                depth2d<float> d [[texture(4)]],
                                texture2d_array<float> arr [[texture(5)]],
                                array<texture2d<float>, 4> many [[texture(6)]],
                                texturecube<float> cube [[texture(10)]],
                                texture_buffer<float, access::read> tb [[texture(11)]],
                                texture2d_ms<float> ms [[texture(12)]],
                                texture3d<int> t3 [[texture(13)]],
                                texture1d<float> t1 [[texture(14)]],
                                depth2d_array<float> darr [[texture(15)]],
                                texture2d<float> unused_tex [[texture(16)]],
                                sampler smp [[sampler(0)]],
                                device float4 *out [[buffer(0)]],
                                uint2 gid [[thread_position_in_grid]]) {
    float2 c = float2(0.5);
    float4 v = r.read(gid) + s.sample(smp, c, level(0)) + float4(d.sample(smp, c, level(0)))
             + arr.sample(smp, c, 0, level(0)) + many[1].sample(smp, c, level(0))
             + cube.sample(smp, float3(1.0, 0.0, 0.0), level(0)) + tb.read(gid.x)
             + ms.read(gid, 0) + float4(t3.read(uint3(gid, 0))) + t1.sample(smp, 0.5)
             + float4(darr.sample(smp, c, 0, level(0)));
    w.write(half4(v), gid);
    rw.write(rw.read(gid) + 1, gid);
    out[gid.x] = v;
}
