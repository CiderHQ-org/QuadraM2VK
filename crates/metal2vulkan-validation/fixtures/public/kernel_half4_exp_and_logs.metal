// `air.exp.v4f16`, `air.log.v4f16` and `air.log2.v4f16` had no authored case and are the highest
// corpus reach left in the coverage queue -- 360 sources call all three, and every one of them is a
// 262KB-plus `cnnConvArray_*` kernel whose function constants have to collapse before it can be
// authored at all. A fixture asks the same three symbols directly.
//
// Half exp/log is byte-authorable and this is why: probed over all 65536 half bit patterns on this
// GPU, `precise::log`, `precise::log2` and `precise::exp` at half width are bit-identical both to
// their `fast::` spellings and to widen-compute-narrow through float. So whichever variant a
// runtime's MSL compiler reaches for -- the question that makes the half POW and trig families
// unauthorable -- these three answer the same bits, and the observation is a property of the
// translation rather than of MoltenVK's shader compiler.
//
// Every input is exactly representable as a half, so nothing rounds on the way in and the only
// rounding left is the function's own. The rows are chosen to tell the three functions apart at
// every lane rather than to make the arithmetic easy: at 2.0 the three answer 7.39, 0.6934 and 1.0,
// so swapping any pair of lowerings, or losing a lane of the vector, moves bytes. The exact anchors
// are there on purpose too -- exp(0) = 1, log(1) = 0, log2 of a power of two is its exponent -- and
// they are the lanes that would survive a wrong implementation, which is why they are not the only
// lanes.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_half4_exp_and_logs(
    device const half4* v [[buffer(0)]],
    device uint* out [[buffer(1)]])
{
    half4 e0 = exp(v[0]);   // 0, 1, -1, 2
    half4 e1 = exp(v[1]);   // -0.5, 3, 0.25, -2
    half4 l0 = log(v[2]);   // 1, 2, 0.5, 10
    half4 l1 = log(v[3]);   // 4, 0.25, 100, 3
    half4 b0 = log2(v[4]);  // 1, 2, 0.5, 256
    half4 b1 = log2(v[5]);  // 3, 10, 0.75, 2^-10
    out[0]  = uint(as_type<ushort>(e0.x)); out[1]  = uint(as_type<ushort>(e0.y));
    out[2]  = uint(as_type<ushort>(e0.z)); out[3]  = uint(as_type<ushort>(e0.w));
    out[4]  = uint(as_type<ushort>(e1.x)); out[5]  = uint(as_type<ushort>(e1.y));
    out[6]  = uint(as_type<ushort>(e1.z)); out[7]  = uint(as_type<ushort>(e1.w));
    out[8]  = uint(as_type<ushort>(l0.x)); out[9]  = uint(as_type<ushort>(l0.y));
    out[10] = uint(as_type<ushort>(l0.z)); out[11] = uint(as_type<ushort>(l0.w));
    out[12] = uint(as_type<ushort>(l1.x)); out[13] = uint(as_type<ushort>(l1.y));
    out[14] = uint(as_type<ushort>(l1.z)); out[15] = uint(as_type<ushort>(l1.w));
    out[16] = uint(as_type<ushort>(b0.x)); out[17] = uint(as_type<ushort>(b0.y));
    out[18] = uint(as_type<ushort>(b0.z)); out[19] = uint(as_type<ushort>(b0.w));
    out[20] = uint(as_type<ushort>(b1.x)); out[21] = uint(as_type<ushort>(b1.y));
    out[22] = uint(as_type<ushort>(b1.z)); out[23] = uint(as_type<ushort>(b1.w));
}
