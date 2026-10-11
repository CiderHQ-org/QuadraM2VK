#include <metal_stdlib>
using namespace metal;

kernel void kernel_bfloat_convert_rounding(
    device const uint    *u32in [[buffer(0)]],
    device const ulong   *u64in [[buffer(1)]],
    device const int4    *i4in  [[buffer(2)]],
    device const ushort4 *bqin  [[buffer(3)]],
    device uint          *out   [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    bfloat4 bq = as_type<bfloat4>(bqin[tid]);
    bfloat  bv = bq.x;
    bfloat  bn = -bv;
    int4    c  = i4in[tid];

    bfloat  wu = bfloat(u32in[tid]);
    bfloat  wl = bfloat(u64in[tid]);
    bfloat4 ws = bfloat4(c);
    bfloat4 wv = bfloat4(uint4(c));
    bfloat  wh = bfloat(ushort(u32in[tid] & 0xffffu));
    int4    ci = int4(bq);

    uint b = tid * 20;
    out[b + 0] = uint(as_type<ushort>(wu));
    out[b + 1] = uint(as_type<ushort>(wl));
    out[b + 2] = uint(as_type<ushort>(ws.x));
    out[b + 3] = uint(as_type<ushort>(ws.y));
    out[b + 4] = uint(as_type<ushort>(ws.z));
    out[b + 5] = uint(as_type<ushort>(ws.w));
    out[b + 6] = uint(as_type<ushort>(wv.x));
    out[b + 7] = uint(as_type<ushort>(wv.y));
    out[b + 8] = uint(as_type<ushort>(wv.z));
    out[b + 9] = uint(as_type<ushort>(wv.w));
    out[b + 10] = uint(ulong(long(bn)) & 0xffffffffull);
    out[b + 11] = uint(ulong(bv) & 0xffffffffull);
    out[b + 12] = uint(bv);
    out[b + 13] = uint(ushort(bv));
    out[b + 14] = uint(uchar(bv));
    out[b + 15] = as_type<uint>(ci.x);
    out[b + 16] = as_type<uint>(ci.y);
    out[b + 17] = as_type<uint>(ci.z);
    out[b + 18] = as_type<uint>(ci.w);
    out[b + 19] = uint(as_type<ushort>(wh));
}
