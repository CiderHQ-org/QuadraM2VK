// Fifteen `air.*` texture symbols the corpus reaches once or twice each and that no authored case
// covered: read_texture_2d_ms.v4f16, read_texture_2d_ms_array.{v4f32,u.v4i16},
// read_texture_2d_array.u.v4i16, gather_texture_2d.{s.v4i32,s.v4i16,u.v4i16},
// gather_texture_2d_array.s.v4i32, sample_texture_3d.{u.v4i32,s.v4i32}, sample_texture_2d.s.v4i16,
// read_texture_buffer_1d.v4f16, write_texture_buffer_1d.v4f16,
// write_texture_2d_array.i16.s.v4i32 and fence_texture_2d. One thread reads or writes each once.
//
// Every texel is a distinct small integer -- exact in half, in float and in both 16- and 32-bit
// integers -- so no row's answer depends on a rounding mode and what this fixture can see is a
// wrong TEXEL, a wrong SAMPLE or a wrong LAYER. Each value encodes its own address: layer, row,
// column, sample index and component all contribute a different digit.
//
// The sampler is a `constexpr` `filter::nearest` one, both because the integer formats here cannot
// be filtered and because it keeps the sampler state out of the case. The gather/sample coordinate
// is 0.5625, which is 2.25 texels into a four-wide texture: `sample` lands on column 2 with no tie
// to break, and the gather footprint brackets columns 1 and 2 at weight 0.75, again with no tie.
// Each of the four gather rows reads a DIFFERENT component of the result, which pins Metal's
// (x0,y1), (x1,y1), (x1,y0), (x0,y0) order rather than just the footprint.
//
// The `read_write` texture is the reason `fence_texture_2d` is here at all: the thread writes it,
// fences, and reads its own write back, which is exactly what the fence orders.
#include <metal_stdlib>
using namespace metal;

constexpr sampler nearest_clamp(coord::normalized, address::clamp_to_edge,
                                filter::nearest, mip_filter::none);

kernel void kernel_texture_variant_leftovers(
    texture2d_ms<half> msh [[texture(0)]],
    texture2d_ms_array<float> msaf [[texture(1)]],
    texture2d_ms_array<ushort> msau [[texture(2)]],
    texture2d_array<ushort> arru [[texture(3)]],
    texture2d<int> ti [[texture(4)]],
    texture2d<short> ts [[texture(5)]],
    texture2d<ushort> tu [[texture(6)]],
    texture2d_array<int> tai [[texture(7)]],
    texture3d<uint> t3u [[texture(8)]],
    texture3d<int> t3i [[texture(9)]],
    texture_buffer<half, access::read> tbr [[texture(10)]],
    texture_buffer<half, access::write> tbw [[texture(11)]],
    texture2d_array<int, access::write> taw [[texture(12)]],
    texture2d<float, access::read_write> trw [[texture(13)]],
    device uint* out [[buffer(0)]])
{
    // The gather/sample coordinate. 0.5625 * 4 = 2.25, so the nearest texel is column 2 and the
    // gather footprint brackets columns 1 and 2 with no tie to break.
    const float2 c = float2(0.5625, 0.5625);
    const float3 c3 = float3(0.75, 0.25, 0.75);

    out[0] = uint(as_type<ushort>(msh.read(uint2(1, 0), 1).x));
    out[1] = as_type<uint>(msaf.read(uint2(0, 0), 1, 2).y);
    out[2] = uint(msau.read(uint2(0, 0), 1, 3).z);
    out[3] = uint(arru.read(uint2(1, 1), 1).w);

    out[4] = as_type<uint>(ti.gather(nearest_clamp, c).x);
    out[5] = uint(as_type<ushort>(ts.gather(nearest_clamp, c).y));
    out[6] = uint(tu.gather(nearest_clamp, c).z);
    out[7] = as_type<uint>(tai.gather(nearest_clamp, c, 0).w);

    out[8] = t3u.sample(nearest_clamp, c3).x;
    out[9] = as_type<uint>(t3i.sample(nearest_clamp, c3).y);
    out[10] = uint(as_type<ushort>(ts.sample(nearest_clamp, c).z));

    out[11] = uint(as_type<ushort>(tbr.read(uint(2)).x));

    tbw.write(half4(0.5h, -1.5h, 8.0h, 0.25h), uint(1));
    taw.write(int4(-7, 11, 0, 2147483647), ushort2(1, 0), 0);

    // A read_write texture: the write, the fence that orders it against this thread's own read,
    // and the read back.
    trw.write(float4(1.5, -2.25, 0.0, 64.0), uint2(0, 0));
    trw.fence();
    float4 back = trw.read(uint2(0, 0));
    out[12] = as_type<uint>(back.x);
    out[13] = as_type<uint>(back.w);
}
