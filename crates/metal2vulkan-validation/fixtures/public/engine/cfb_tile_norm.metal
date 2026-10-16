// Owned fixture: tile functions over a normalized color attachment's implicit imageblock, as Metal
// probes ran them on the M3 Max and an M4: a member of one, two or four floats, each thread reading
// its cell into its pixel's element of a buffer, or writing the element into its cell, and one
// function writing its cell and reading it back after a barrier; and, as probe 12 ran them, a
// member of one, two or four halves, read into a float buffer and written from a half buffer. A
// tile is 16x16 pixels and its threadgroup 16x16 threads, over an attachment one tile wide, so
// pixel `(x, y)` is element `16 y + x`. Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct CfbTn1 {
    float c [[color(0)]];
};

struct CfbTn2 {
    float2 c [[color(0)]];
};

struct CfbTn4 {
    float4 c [[color(0)]];
};

static uint cfb_tn_index(uint2 tg, ushort2 lid) {
    return (tg.y * 16u + uint(lid.y)) * 16u + uint(lid.x);
}

kernel void cfb_tn_read1(imageblock<CfbTn1, imageblock_layout_implicit> ib,
                         device float4 *out [[buffer(0)]],
                         ushort2 lid [[thread_position_in_threadgroup]],
                         uint2 tg [[threadgroup_position_in_grid]]) {
    out[cfb_tn_index(tg, lid)] = float4(ib.read(lid).c, -7.0, -7.0, -7.0);
}

kernel void cfb_tn_write1(imageblock<CfbTn1, imageblock_layout_implicit> ib,
                          device const float4 *in [[buffer(0)]],
                          ushort2 lid [[thread_position_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    CfbTn1 v;
    v.c = in[cfb_tn_index(tg, lid)].x;
    ib.write(v, lid);
}

kernel void cfb_tn_read2(imageblock<CfbTn2, imageblock_layout_implicit> ib,
                         device float4 *out [[buffer(0)]],
                         ushort2 lid [[thread_position_in_threadgroup]],
                         uint2 tg [[threadgroup_position_in_grid]]) {
    out[cfb_tn_index(tg, lid)] = float4(ib.read(lid).c, -7.0, -7.0);
}

kernel void cfb_tn_write2(imageblock<CfbTn2, imageblock_layout_implicit> ib,
                          device const float4 *in [[buffer(0)]],
                          ushort2 lid [[thread_position_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    CfbTn2 v;
    v.c = in[cfb_tn_index(tg, lid)].xy;
    ib.write(v, lid);
}

kernel void cfb_tn_read4(imageblock<CfbTn4, imageblock_layout_implicit> ib,
                         device float4 *out [[buffer(0)]],
                         ushort2 lid [[thread_position_in_threadgroup]],
                         uint2 tg [[threadgroup_position_in_grid]]) {
    out[cfb_tn_index(tg, lid)] = ib.read(lid).c;
}

kernel void cfb_tn_write4(imageblock<CfbTn4, imageblock_layout_implicit> ib,
                          device const float4 *in [[buffer(0)]],
                          ushort2 lid [[thread_position_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    CfbTn4 v;
    v.c = in[cfb_tn_index(tg, lid)];
    ib.write(v, lid);
}

// Writes the element into the cell, then reads the cell back into the element.
kernel void cfb_tn_round4(imageblock<CfbTn4, imageblock_layout_implicit> ib,
                          device float4 *io [[buffer(0)]],
                          ushort2 lid [[thread_position_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    uint i = cfb_tn_index(tg, lid);
    CfbTn4 v;
    v.c = io[i];
    ib.write(v, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    io[i] = ib.read(lid).c;
}

struct CfbTnH1 {
    half c [[color(0)]];
};

struct CfbTnH2 {
    half2 c [[color(0)]];
};

struct CfbTnH4 {
    half4 c [[color(0)]];
};

kernel void cfb_tn_hread1(imageblock<CfbTnH1, imageblock_layout_implicit> ib,
                          device float4 *out [[buffer(0)]],
                          ushort2 lid [[thread_position_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    out[cfb_tn_index(tg, lid)] = float4(float(ib.read(lid).c), -7.0, -7.0, -7.0);
}

kernel void cfb_tn_hwrite1(imageblock<CfbTnH1, imageblock_layout_implicit> ib,
                           device const half4 *in [[buffer(0)]],
                           ushort2 lid [[thread_position_in_threadgroup]],
                           uint2 tg [[threadgroup_position_in_grid]]) {
    CfbTnH1 v;
    v.c = in[cfb_tn_index(tg, lid)].x;
    ib.write(v, lid);
}

kernel void cfb_tn_hread2(imageblock<CfbTnH2, imageblock_layout_implicit> ib,
                          device float4 *out [[buffer(0)]],
                          ushort2 lid [[thread_position_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    out[cfb_tn_index(tg, lid)] = float4(float2(ib.read(lid).c), -7.0, -7.0);
}

kernel void cfb_tn_hwrite2(imageblock<CfbTnH2, imageblock_layout_implicit> ib,
                           device const half4 *in [[buffer(0)]],
                           ushort2 lid [[thread_position_in_threadgroup]],
                           uint2 tg [[threadgroup_position_in_grid]]) {
    CfbTnH2 v;
    v.c = in[cfb_tn_index(tg, lid)].xy;
    ib.write(v, lid);
}

kernel void cfb_tn_hread4(imageblock<CfbTnH4, imageblock_layout_implicit> ib,
                          device float4 *out [[buffer(0)]],
                          ushort2 lid [[thread_position_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    out[cfb_tn_index(tg, lid)] = float4(ib.read(lid).c);
}

kernel void cfb_tn_hwrite4(imageblock<CfbTnH4, imageblock_layout_implicit> ib,
                           device const half4 *in [[buffer(0)]],
                           ushort2 lid [[thread_position_in_threadgroup]],
                           uint2 tg [[threadgroup_position_in_grid]]) {
    CfbTnH4 v;
    v.c = in[cfb_tn_index(tg, lid)];
    ib.write(v, lid);
}
