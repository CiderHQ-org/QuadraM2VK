#include <metal_stdlib>
using namespace metal;

kernel void kernel_fast_inverse_trig(
    device uint *out [[buffer(0)]],
    device const float *in [[buffer(1)]])
{
    out[0]  = as_type<uint>(fast::tan(in[0]));
    out[1]  = as_type<uint>(fast::tan(in[1]));
    out[2]  = as_type<uint>(fast::tan(in[2]));
    out[3]  = as_type<uint>(fast::tan(in[3]));
    out[4]  = as_type<uint>(fast::tan(in[4]));
    out[5]  = as_type<uint>(fast::tan(in[5]));
    out[6]  = as_type<uint>(fast::tan(in[6]));
    out[7]  = as_type<uint>(fast::tan(in[7]));
    out[8]  = as_type<uint>(fast::tan(in[8]));
    out[9]  = as_type<uint>(fast::tan(in[9]));
    out[10] = as_type<uint>(fast::tan(in[10]));
    out[11] = as_type<uint>(fast::tan(in[11]));
    out[12] = as_type<uint>(fast::atan(in[12]));
    out[13] = as_type<uint>(fast::atan(in[13]));
    out[14] = as_type<uint>(fast::atan(in[14]));
    out[15] = as_type<uint>(fast::atan(in[15]));
    out[16] = as_type<uint>(fast::atan(in[16]));
    out[17] = as_type<uint>(fast::atan(in[17]));
    out[18] = as_type<uint>(fast::atan(in[18]));
    out[19] = as_type<uint>(fast::atan(in[19]));
    out[20] = as_type<uint>(fast::atan(in[20]));
    out[21] = as_type<uint>(fast::atan(in[21]));
    out[22] = as_type<uint>(fast::atan(in[22]));
    out[23] = as_type<uint>(fast::atan(in[23]));
    out[24] = as_type<uint>(fast::asin(in[24]));
    out[25] = as_type<uint>(fast::asin(in[25]));
    out[26] = as_type<uint>(fast::asin(in[26]));
    out[27] = as_type<uint>(fast::asin(in[27]));
    out[28] = as_type<uint>(fast::asin(in[28]));
    out[29] = as_type<uint>(fast::asin(in[29]));
    out[30] = as_type<uint>(fast::asin(in[30]));
    out[31] = as_type<uint>(fast::asin(in[31]));
    out[32] = as_type<uint>(fast::asin(in[32]));
    out[33] = as_type<uint>(fast::asin(in[33]));
    out[34] = as_type<uint>(fast::asin(in[34]));
    out[35] = as_type<uint>(fast::asin(in[35]));
}
