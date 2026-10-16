// Owned fixture (lower B2): texture reads, writes, fences, and
// queries through static texture slots, one kernel per shape family. Loop-free.
//
// - texel_buffers: a uniform texel buffer read and a storage texel buffer write.
// - read_write_fence: a read_write texture read, written, fenced, and read again.
// - cube_faces: cube and cube-array reads by face (the binder's 2D-array view), cube queries.
// - queries: size, array-size, level, and sample-count queries, and is_null_texture.
// - short_texels: 16-bit coordinates, 16-bit signed texels read and written, a half 3D read.
// - depth_ms: a depth read and a multisampled integer read.
// - write_lod: a write at a dynamic level, through the slot's level views.
#include <metal_stdlib>
using namespace metal;

kernel void texel_buffers(texture_buffer<float, access::read> src [[texture(0)]],
                          texture_buffer<float, access::write> dst [[texture(1)]],
                          uint gid [[thread_position_in_grid]]) {
    dst.write(src.read(gid) * 2.0, gid);
}

kernel void read_write_fence(texture2d<uint, access::read_write> t [[texture(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint4 v = t.read(gid);
    t.write(v + uint4(1, 2, 3, 4), gid);
    t.fence();
    uint4 w = t.read(gid);
    t.write(w * 2, gid);
}

kernel void cube_faces(texturecube<float, access::read> c [[texture(0)]],
                       texturecube_array<float, access::read> ca [[texture(1)]],
                       device float4 *out [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    out[0] = c.read(gid, 3, 0);
    out[1] = ca.read(gid, 4, 1, 0);
    out[2] = float4(c.get_width(), ca.get_array_size(), c.get_num_mip_levels(), 0);
}

kernel void queries(texture2d_array<float, access::sample> a [[texture(0)]],
                    texture3d<half, access::read> v [[texture(1)]],
                    texture2d_ms<float, access::read> ms [[texture(2)]],
                    texture2d<float, access::write> w [[texture(3)]],
                    device uint *out [[buffer(0)]]) {
    out[0] = a.get_width(1);
    out[1] = a.get_height(1);
    out[2] = a.get_array_size();
    out[3] = a.get_num_mip_levels();
    out[4] = v.get_depth(0);
    out[5] = ms.get_num_samples();
    out[6] = ms.get_width();
    out[7] = w.get_width();
    out[8] = is_null_texture(w) ? 1 : 0;
}

kernel void short_texels(texture2d<short, access::read> s [[texture(0)]],
                         texture2d<short, access::write> d [[texture(1)]],
                         texture3d<half, access::read> h [[texture(2)]],
                         device half4 *out [[buffer(0)]],
                         ushort2 gid [[thread_position_in_grid]]) {
    short4 x = s.read(gid, 0);
    d.write(x - short4(1), gid);
    out[0] = h.read(ushort3(gid, 0), 1);
}

kernel void depth_ms(depth2d<float, access::read> dp [[texture(0)]],
                     texture2d_ms<int, access::read> ms [[texture(1)]],
                     device float *out [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
    out[0] = dp.read(gid, 0);
    out[1] = float(ms.read(gid, 2).y);
}

kernel void write_lod(texture2d<float, access::write> t [[texture(0)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint l [[thread_index_in_threadgroup]]) {
    t.write(float4(1.0), gid, l);
}
