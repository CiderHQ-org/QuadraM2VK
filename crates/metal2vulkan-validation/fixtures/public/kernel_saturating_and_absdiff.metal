#include <metal_stdlib>
using namespace metal;

kernel void kernel_saturating_and_absdiff(
    device uint *out [[buffer(0)]],
    device const uchar *b8 [[buffer(1)]],
    device const ushort *b16 [[buffer(2)]],
    device const uint *b32 [[buffer(3)]])
{
    uchar3 p = uchar3(b8[0], b8[1], b8[2]);
    uchar3 q = uchar3(b8[3], b8[4], b8[5]);
    uchar3 sadd = addsat(p, q);
    uchar3 ssub = subsat(p, q);

    ushort4 r = ushort4(b16[0], b16[1], b16[2], b16[3]);
    ushort4 s = ushort4(b16[4], b16[5], b16[6], b16[7]);
    ushort4 ad4 = absdiff(r, s);
    ushort2 ad2 = absdiff(ushort2(b16[0], b16[1]), ushort2(b16[4], b16[5]));
    ushort rh = rhadd(b16[8], b16[9]);

    uint adu = absdiff(b32[0], b32[1]);
    int4 si = int4(as_type<int>(b32[2]), as_type<int>(b32[3]), as_type<int>(b32[4]), as_type<int>(b32[5]));
    int4 sj = int4(as_type<int>(b32[6]), as_type<int>(b32[7]), as_type<int>(b32[8]), as_type<int>(b32[9]));
    uint4 ads = absdiff(si, sj);
    int ms = madsat(as_type<int>(b32[10]), as_type<int>(b32[11]), as_type<int>(b32[12]));

    out[0] = uint(sadd.x); out[1] = uint(sadd.y); out[2] = uint(sadd.z);
    out[3] = uint(ssub.x); out[4] = uint(ssub.y); out[5] = uint(ssub.z);
    out[6] = uint(ad4.x); out[7] = uint(ad4.y); out[8] = uint(ad4.z); out[9] = uint(ad4.w);
    out[10] = uint(ad2.x); out[11] = uint(ad2.y);
    out[12] = uint(rh);
    out[13] = adu;
    out[14] = ads.x; out[15] = ads.y;
    out[16] = ads.z; out[17] = ads.w;
    out[18] = as_type<uint>(ms);
}
