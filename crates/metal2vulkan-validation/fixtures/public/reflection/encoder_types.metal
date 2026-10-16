// Owned fixture, generated: one struct per MSL
// scalar, vector and matrix type that Metal's MTLDataType names, each held alone by a plain buffer
// (`encoder_types_plain_<n>`) and after a texture by an argument buffer (`encoder_types_tail_<n>`),
// so Metal's `newArgumentEncoderWithBufferBinding:` states each type's bytes and alignment in
// both. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct EncT_float { float a; };
struct EncTail_float { texture2d<float> t; float a; };
struct EncT_float2 { float2 a; };
struct EncTail_float2 { texture2d<float> t; float2 a; };
struct EncT_float3 { float3 a; };
struct EncTail_float3 { texture2d<float> t; float3 a; };
struct EncT_float4 { float4 a; };
struct EncTail_float4 { texture2d<float> t; float4 a; };
struct EncT_float2X2 { float2x2 a; };
struct EncTail_float2X2 { texture2d<float> t; float2x2 a; };
struct EncT_float2X3 { float2x3 a; };
struct EncTail_float2X3 { texture2d<float> t; float2x3 a; };
struct EncT_float2X4 { float2x4 a; };
struct EncTail_float2X4 { texture2d<float> t; float2x4 a; };
struct EncT_float3X2 { float3x2 a; };
struct EncTail_float3X2 { texture2d<float> t; float3x2 a; };
struct EncT_float3X3 { float3x3 a; };
struct EncTail_float3X3 { texture2d<float> t; float3x3 a; };
struct EncT_float3X4 { float3x4 a; };
struct EncTail_float3X4 { texture2d<float> t; float3x4 a; };
struct EncT_float4X2 { float4x2 a; };
struct EncTail_float4X2 { texture2d<float> t; float4x2 a; };
struct EncT_float4X3 { float4x3 a; };
struct EncTail_float4X3 { texture2d<float> t; float4x3 a; };
struct EncT_float4X4 { float4x4 a; };
struct EncTail_float4X4 { texture2d<float> t; float4x4 a; };
struct EncT_half { half a; };
struct EncTail_half { texture2d<float> t; half a; };
struct EncT_half2 { half2 a; };
struct EncTail_half2 { texture2d<float> t; half2 a; };
struct EncT_half3 { half3 a; };
struct EncTail_half3 { texture2d<float> t; half3 a; };
struct EncT_half4 { half4 a; };
struct EncTail_half4 { texture2d<float> t; half4 a; };
struct EncT_half2X2 { half2x2 a; };
struct EncTail_half2X2 { texture2d<float> t; half2x2 a; };
struct EncT_half2X3 { half2x3 a; };
struct EncTail_half2X3 { texture2d<float> t; half2x3 a; };
struct EncT_half2X4 { half2x4 a; };
struct EncTail_half2X4 { texture2d<float> t; half2x4 a; };
struct EncT_half3X2 { half3x2 a; };
struct EncTail_half3X2 { texture2d<float> t; half3x2 a; };
struct EncT_half3X3 { half3x3 a; };
struct EncTail_half3X3 { texture2d<float> t; half3x3 a; };
struct EncT_half3X4 { half3x4 a; };
struct EncTail_half3X4 { texture2d<float> t; half3x4 a; };
struct EncT_half4X2 { half4x2 a; };
struct EncTail_half4X2 { texture2d<float> t; half4x2 a; };
struct EncT_half4X3 { half4x3 a; };
struct EncTail_half4X3 { texture2d<float> t; half4x3 a; };
struct EncT_half4X4 { half4x4 a; };
struct EncTail_half4X4 { texture2d<float> t; half4x4 a; };
struct EncT_bfloat { bfloat a; };
struct EncTail_bfloat { texture2d<float> t; bfloat a; };
struct EncT_bfloat2 { bfloat2 a; };
struct EncTail_bfloat2 { texture2d<float> t; bfloat2 a; };
struct EncT_bfloat3 { bfloat3 a; };
struct EncTail_bfloat3 { texture2d<float> t; bfloat3 a; };
struct EncT_bfloat4 { bfloat4 a; };
struct EncTail_bfloat4 { texture2d<float> t; bfloat4 a; };
struct EncT_int { int a; };
struct EncTail_int { texture2d<float> t; int a; };
struct EncT_int2 { int2 a; };
struct EncTail_int2 { texture2d<float> t; int2 a; };
struct EncT_int3 { int3 a; };
struct EncTail_int3 { texture2d<float> t; int3 a; };
struct EncT_int4 { int4 a; };
struct EncTail_int4 { texture2d<float> t; int4 a; };
struct EncT_uint { uint a; };
struct EncTail_uint { texture2d<float> t; uint a; };
struct EncT_uint2 { uint2 a; };
struct EncTail_uint2 { texture2d<float> t; uint2 a; };
struct EncT_uint3 { uint3 a; };
struct EncTail_uint3 { texture2d<float> t; uint3 a; };
struct EncT_uint4 { uint4 a; };
struct EncTail_uint4 { texture2d<float> t; uint4 a; };
struct EncT_short { short a; };
struct EncTail_short { texture2d<float> t; short a; };
struct EncT_short2 { short2 a; };
struct EncTail_short2 { texture2d<float> t; short2 a; };
struct EncT_short3 { short3 a; };
struct EncTail_short3 { texture2d<float> t; short3 a; };
struct EncT_short4 { short4 a; };
struct EncTail_short4 { texture2d<float> t; short4 a; };
struct EncT_ushort { ushort a; };
struct EncTail_ushort { texture2d<float> t; ushort a; };
struct EncT_ushort2 { ushort2 a; };
struct EncTail_ushort2 { texture2d<float> t; ushort2 a; };
struct EncT_ushort3 { ushort3 a; };
struct EncTail_ushort3 { texture2d<float> t; ushort3 a; };
struct EncT_ushort4 { ushort4 a; };
struct EncTail_ushort4 { texture2d<float> t; ushort4 a; };
struct EncT_char { char a; };
struct EncTail_char { texture2d<float> t; char a; };
struct EncT_char2 { char2 a; };
struct EncTail_char2 { texture2d<float> t; char2 a; };
struct EncT_char3 { char3 a; };
struct EncTail_char3 { texture2d<float> t; char3 a; };
struct EncT_char4 { char4 a; };
struct EncTail_char4 { texture2d<float> t; char4 a; };
struct EncT_uchar { uchar a; };
struct EncTail_uchar { texture2d<float> t; uchar a; };
struct EncT_uchar2 { uchar2 a; };
struct EncTail_uchar2 { texture2d<float> t; uchar2 a; };
struct EncT_uchar3 { uchar3 a; };
struct EncTail_uchar3 { texture2d<float> t; uchar3 a; };
struct EncT_uchar4 { uchar4 a; };
struct EncTail_uchar4 { texture2d<float> t; uchar4 a; };
struct EncT_bool { bool a; };
struct EncTail_bool { texture2d<float> t; bool a; };
struct EncT_bool2 { bool2 a; };
struct EncTail_bool2 { texture2d<float> t; bool2 a; };
struct EncT_bool3 { bool3 a; };
struct EncTail_bool3 { texture2d<float> t; bool3 a; };
struct EncT_bool4 { bool4 a; };
struct EncTail_bool4 { texture2d<float> t; bool4 a; };
struct EncT_long { long a; };
struct EncTail_long { texture2d<float> t; long a; };
struct EncT_long2 { long2 a; };
struct EncTail_long2 { texture2d<float> t; long2 a; };
struct EncT_long3 { long3 a; };
struct EncTail_long3 { texture2d<float> t; long3 a; };
struct EncT_long4 { long4 a; };
struct EncTail_long4 { texture2d<float> t; long4 a; };
struct EncT_ulong { ulong a; };
struct EncTail_ulong { texture2d<float> t; ulong a; };
struct EncT_ulong2 { ulong2 a; };
struct EncTail_ulong2 { texture2d<float> t; ulong2 a; };
struct EncT_ulong3 { ulong3 a; };
struct EncTail_ulong3 { texture2d<float> t; ulong3 a; };
struct EncT_ulong4 { ulong4 a; };
struct EncTail_ulong4 { texture2d<float> t; ulong4 a; };

kernel void encoder_types_plain_0(
    device EncT_float *b0 [[buffer(0)]],
    device EncT_float2 *b1 [[buffer(1)]],
    device EncT_float3 *b2 [[buffer(2)]],
    device EncT_float4 *b3 [[buffer(3)]],
    device EncT_float2X2 *b4 [[buffer(4)]],
    device EncT_float2X3 *b5 [[buffer(5)]],
    device EncT_float2X4 *b6 [[buffer(6)]],
    device EncT_float3X2 *b7 [[buffer(7)]],
    device EncT_float3X3 *b8 [[buffer(8)]],
    device EncT_float3X4 *b9 [[buffer(9)]],
    device EncT_float4X2 *b10 [[buffer(10)]],
    device EncT_float4X3 *b11 [[buffer(11)]],
    device EncT_float4X4 *b12 [[buffer(12)]],
    device EncT_half *b13 [[buffer(13)]],
    device EncT_half2 *b14 [[buffer(14)]],
    device EncT_half3 *b15 [[buffer(15)]],
    device EncT_half4 *b16 [[buffer(16)]],
    device EncT_half2X2 *b17 [[buffer(17)]],
    device EncT_half2X3 *b18 [[buffer(18)]],
    device EncT_half2X4 *b19 [[buffer(19)]],
    device EncT_half3X2 *b20 [[buffer(20)]],
    device EncT_half3X3 *b21 [[buffer(21)]],
    uint tid [[thread_position_in_grid]]) {
    b0[tid].a = float();
    b1[tid].a = float2();
    b2[tid].a = float3();
    b3[tid].a = float4();
    b4[tid].a = float2x2();
    b5[tid].a = float2x3();
    b6[tid].a = float2x4();
    b7[tid].a = float3x2();
    b8[tid].a = float3x3();
    b9[tid].a = float3x4();
    b10[tid].a = float4x2();
    b11[tid].a = float4x3();
    b12[tid].a = float4x4();
    b13[tid].a = half();
    b14[tid].a = half2();
    b15[tid].a = half3();
    b16[tid].a = half4();
    b17[tid].a = half2x2();
    b18[tid].a = half2x3();
    b19[tid].a = half2x4();
    b20[tid].a = half3x2();
    b21[tid].a = half3x3();
}

kernel void encoder_types_plain_1(
    device EncT_half3X4 *b0 [[buffer(0)]],
    device EncT_half4X2 *b1 [[buffer(1)]],
    device EncT_half4X3 *b2 [[buffer(2)]],
    device EncT_half4X4 *b3 [[buffer(3)]],
    device EncT_bfloat *b4 [[buffer(4)]],
    device EncT_bfloat2 *b5 [[buffer(5)]],
    device EncT_bfloat3 *b6 [[buffer(6)]],
    device EncT_bfloat4 *b7 [[buffer(7)]],
    device EncT_int *b8 [[buffer(8)]],
    device EncT_int2 *b9 [[buffer(9)]],
    device EncT_int3 *b10 [[buffer(10)]],
    device EncT_int4 *b11 [[buffer(11)]],
    device EncT_uint *b12 [[buffer(12)]],
    device EncT_uint2 *b13 [[buffer(13)]],
    device EncT_uint3 *b14 [[buffer(14)]],
    device EncT_uint4 *b15 [[buffer(15)]],
    device EncT_short *b16 [[buffer(16)]],
    device EncT_short2 *b17 [[buffer(17)]],
    device EncT_short3 *b18 [[buffer(18)]],
    device EncT_short4 *b19 [[buffer(19)]],
    device EncT_ushort *b20 [[buffer(20)]],
    device EncT_ushort2 *b21 [[buffer(21)]],
    uint tid [[thread_position_in_grid]]) {
    b0[tid].a = half3x4();
    b1[tid].a = half4x2();
    b2[tid].a = half4x3();
    b3[tid].a = half4x4();
    b4[tid].a = bfloat();
    b5[tid].a = bfloat2();
    b6[tid].a = bfloat3();
    b7[tid].a = bfloat4();
    b8[tid].a = int();
    b9[tid].a = int2();
    b10[tid].a = int3();
    b11[tid].a = int4();
    b12[tid].a = uint();
    b13[tid].a = uint2();
    b14[tid].a = uint3();
    b15[tid].a = uint4();
    b16[tid].a = short();
    b17[tid].a = short2();
    b18[tid].a = short3();
    b19[tid].a = short4();
    b20[tid].a = ushort();
    b21[tid].a = ushort2();
}

kernel void encoder_types_plain_2(
    device EncT_ushort3 *b0 [[buffer(0)]],
    device EncT_ushort4 *b1 [[buffer(1)]],
    device EncT_char *b2 [[buffer(2)]],
    device EncT_char2 *b3 [[buffer(3)]],
    device EncT_char3 *b4 [[buffer(4)]],
    device EncT_char4 *b5 [[buffer(5)]],
    device EncT_uchar *b6 [[buffer(6)]],
    device EncT_uchar2 *b7 [[buffer(7)]],
    device EncT_uchar3 *b8 [[buffer(8)]],
    device EncT_uchar4 *b9 [[buffer(9)]],
    device EncT_bool *b10 [[buffer(10)]],
    device EncT_bool2 *b11 [[buffer(11)]],
    device EncT_bool3 *b12 [[buffer(12)]],
    device EncT_bool4 *b13 [[buffer(13)]],
    device EncT_long *b14 [[buffer(14)]],
    device EncT_long2 *b15 [[buffer(15)]],
    device EncT_long3 *b16 [[buffer(16)]],
    device EncT_long4 *b17 [[buffer(17)]],
    device EncT_ulong *b18 [[buffer(18)]],
    device EncT_ulong2 *b19 [[buffer(19)]],
    device EncT_ulong3 *b20 [[buffer(20)]],
    device EncT_ulong4 *b21 [[buffer(21)]],
    uint tid [[thread_position_in_grid]]) {
    b0[tid].a = ushort3();
    b1[tid].a = ushort4();
    b2[tid].a = char();
    b3[tid].a = char2();
    b4[tid].a = char3();
    b5[tid].a = char4();
    b6[tid].a = uchar();
    b7[tid].a = uchar2();
    b8[tid].a = uchar3();
    b9[tid].a = uchar4();
    b10[tid].a = bool();
    b11[tid].a = bool2();
    b12[tid].a = bool3();
    b13[tid].a = bool4();
    b14[tid].a = long();
    b15[tid].a = long2();
    b16[tid].a = long3();
    b17[tid].a = long4();
    b18[tid].a = ulong();
    b19[tid].a = ulong2();
    b20[tid].a = ulong3();
    b21[tid].a = ulong4();
}

kernel void encoder_types_tail_0(
    device EncTail_float &a0 [[buffer(0)]],
    device EncTail_float2 &a1 [[buffer(1)]],
    device EncTail_float3 &a2 [[buffer(2)]],
    device EncTail_float4 &a3 [[buffer(3)]],
    device EncTail_float2X2 &a4 [[buffer(4)]],
    device EncTail_float2X3 &a5 [[buffer(5)]],
    device EncTail_float2X4 &a6 [[buffer(6)]],
    device EncTail_float3X2 &a7 [[buffer(7)]],
    device EncTail_float3X3 &a8 [[buffer(8)]],
    device EncTail_float3X4 &a9 [[buffer(9)]],
    device EncTail_float4X2 &a10 [[buffer(10)]],
    device EncTail_float4X3 &a11 [[buffer(11)]],
    device EncTail_float4X4 &a12 [[buffer(12)]],
    device EncTail_half &a13 [[buffer(13)]],
    device EncTail_half2 &a14 [[buffer(14)]],
    device EncTail_half3 &a15 [[buffer(15)]],
    device EncTail_half4 &a16 [[buffer(16)]],
    device EncTail_half2X2 &a17 [[buffer(17)]],
    device EncTail_half2X3 &a18 [[buffer(18)]],
    device EncTail_half2X4 &a19 [[buffer(19)]],
    device EncTail_half3X2 &a20 [[buffer(20)]],
    device EncTail_half3X3 &a21 [[buffer(21)]],
    device float4 *out [[buffer(22)]],
    uint tid [[thread_position_in_grid]]) {
    out[tid] = a0.t.read(uint2(0));
    a0.a = float();
    a1.a = float2();
    a2.a = float3();
    a3.a = float4();
    a4.a = float2x2();
    a5.a = float2x3();
    a6.a = float2x4();
    a7.a = float3x2();
    a8.a = float3x3();
    a9.a = float3x4();
    a10.a = float4x2();
    a11.a = float4x3();
    a12.a = float4x4();
    a13.a = half();
    a14.a = half2();
    a15.a = half3();
    a16.a = half4();
    a17.a = half2x2();
    a18.a = half2x3();
    a19.a = half2x4();
    a20.a = half3x2();
    a21.a = half3x3();
}

kernel void encoder_types_tail_1(
    device EncTail_half3X4 &a0 [[buffer(0)]],
    device EncTail_half4X2 &a1 [[buffer(1)]],
    device EncTail_half4X3 &a2 [[buffer(2)]],
    device EncTail_half4X4 &a3 [[buffer(3)]],
    device EncTail_bfloat &a4 [[buffer(4)]],
    device EncTail_bfloat2 &a5 [[buffer(5)]],
    device EncTail_bfloat3 &a6 [[buffer(6)]],
    device EncTail_bfloat4 &a7 [[buffer(7)]],
    device EncTail_int &a8 [[buffer(8)]],
    device EncTail_int2 &a9 [[buffer(9)]],
    device EncTail_int3 &a10 [[buffer(10)]],
    device EncTail_int4 &a11 [[buffer(11)]],
    device EncTail_uint &a12 [[buffer(12)]],
    device EncTail_uint2 &a13 [[buffer(13)]],
    device EncTail_uint3 &a14 [[buffer(14)]],
    device EncTail_uint4 &a15 [[buffer(15)]],
    device EncTail_short &a16 [[buffer(16)]],
    device EncTail_short2 &a17 [[buffer(17)]],
    device EncTail_short3 &a18 [[buffer(18)]],
    device EncTail_short4 &a19 [[buffer(19)]],
    device EncTail_ushort &a20 [[buffer(20)]],
    device EncTail_ushort2 &a21 [[buffer(21)]],
    device float4 *out [[buffer(22)]],
    uint tid [[thread_position_in_grid]]) {
    out[tid] = a0.t.read(uint2(0));
    a0.a = half3x4();
    a1.a = half4x2();
    a2.a = half4x3();
    a3.a = half4x4();
    a4.a = bfloat();
    a5.a = bfloat2();
    a6.a = bfloat3();
    a7.a = bfloat4();
    a8.a = int();
    a9.a = int2();
    a10.a = int3();
    a11.a = int4();
    a12.a = uint();
    a13.a = uint2();
    a14.a = uint3();
    a15.a = uint4();
    a16.a = short();
    a17.a = short2();
    a18.a = short3();
    a19.a = short4();
    a20.a = ushort();
    a21.a = ushort2();
}

kernel void encoder_types_tail_2(
    device EncTail_ushort3 &a0 [[buffer(0)]],
    device EncTail_ushort4 &a1 [[buffer(1)]],
    device EncTail_char &a2 [[buffer(2)]],
    device EncTail_char2 &a3 [[buffer(3)]],
    device EncTail_char3 &a4 [[buffer(4)]],
    device EncTail_char4 &a5 [[buffer(5)]],
    device EncTail_uchar &a6 [[buffer(6)]],
    device EncTail_uchar2 &a7 [[buffer(7)]],
    device EncTail_uchar3 &a8 [[buffer(8)]],
    device EncTail_uchar4 &a9 [[buffer(9)]],
    device EncTail_bool &a10 [[buffer(10)]],
    device EncTail_bool2 &a11 [[buffer(11)]],
    device EncTail_bool3 &a12 [[buffer(12)]],
    device EncTail_bool4 &a13 [[buffer(13)]],
    device EncTail_long &a14 [[buffer(14)]],
    device EncTail_long2 &a15 [[buffer(15)]],
    device EncTail_long3 &a16 [[buffer(16)]],
    device EncTail_long4 &a17 [[buffer(17)]],
    device EncTail_ulong &a18 [[buffer(18)]],
    device EncTail_ulong2 &a19 [[buffer(19)]],
    device EncTail_ulong3 &a20 [[buffer(20)]],
    device EncTail_ulong4 &a21 [[buffer(21)]],
    device float4 *out [[buffer(22)]],
    uint tid [[thread_position_in_grid]]) {
    out[tid] = a0.t.read(uint2(0));
    a0.a = ushort3();
    a1.a = ushort4();
    a2.a = char();
    a3.a = char2();
    a4.a = char3();
    a5.a = char4();
    a6.a = uchar();
    a7.a = uchar2();
    a8.a = uchar3();
    a9.a = uchar4();
    a10.a = bool();
    a11.a = bool2();
    a12.a = bool3();
    a13.a = bool4();
    a14.a = long();
    a15.a = long2();
    a16.a = long3();
    a17.a = long4();
    a18.a = ulong();
    a19.a = ulong2();
    a20.a = ulong3();
    a21.a = ulong4();
}
