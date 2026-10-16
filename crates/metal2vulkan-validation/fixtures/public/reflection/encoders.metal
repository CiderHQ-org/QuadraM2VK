// Owned fixture: buffers whose reflected bindings
// Metal's `newArgumentEncoderWithBufferBinding:` lays out. `encoders_plain` takes plain buffers of
// one struct each: single members of each scalar, vector, packed vector and matrix type, arrays,
// members in both orders, a nested struct and an array of them, and atomics. `encoders_args` takes
// argument buffers whose last member is a constant of each kind (which decides the encoded
// length), a plain-data struct held by value, an array of them, and explicit ids with a gap.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct EncF { float a; };
struct EncF2 { float2 a; };
struct EncF3 { float3 a; };
struct EncF4 { float4 a; };
struct EncPF3 { packed_float3 a; };
struct EncH { half a; };
struct EncH4 { half4 a; };
struct EncC { char a; };
struct EncS { short a; };
struct EncUL { ulong a; };
struct EncB { bool a; };
struct EncI3 { int3 a; };
struct EncM22 { float2x2 m; };
struct EncM33 { float3x3 m; };
struct EncM44 { float4x4 m; };
struct EncFArr { float a[3]; };
struct EncF4Arr { float4 a[2]; };
struct EncFThenF4 { float a; float4 b; };
struct EncF4ThenF { float4 a; float b; };
struct EncCThenF { char a; float b; };
struct EncNest { EncF2 in; float b; };
struct EncNestArr { EncF2 in[2]; };
struct EncAtomicMember { atomic_uint c; float f; };

kernel void encoders_plain(device EncF *b0 [[buffer(0)]],
                           device EncF2 *b1 [[buffer(1)]],
                           device EncF3 *b2 [[buffer(2)]],
                           device EncF4 *b3 [[buffer(3)]],
                           device EncPF3 *b4 [[buffer(4)]],
                           device EncH *b5 [[buffer(5)]],
                           device EncH4 *b6 [[buffer(6)]],
                           device EncC *b7 [[buffer(7)]],
                           device EncS *b8 [[buffer(8)]],
                           device EncUL *b9 [[buffer(9)]],
                           device EncB *b10 [[buffer(10)]],
                           device EncI3 *b11 [[buffer(11)]],
                           device EncM22 *b12 [[buffer(12)]],
                           device EncM33 *b13 [[buffer(13)]],
                           device EncM44 *b14 [[buffer(14)]],
                           device EncFArr *b15 [[buffer(15)]],
                           device EncF4Arr *b16 [[buffer(16)]],
                           device EncFThenF4 *b17 [[buffer(17)]],
                           device EncF4ThenF *b18 [[buffer(18)]],
                           device EncCThenF *b19 [[buffer(19)]],
                           device EncNest *b20 [[buffer(20)]],
                           device EncNestArr *b21 [[buffer(21)]],
                           device atomic_uint *b22 [[buffer(22)]],
                           device atomic_int *b23 [[buffer(23)]],
                           device EncAtomicMember *b24 [[buffer(24)]],
                           uint tid [[thread_position_in_grid]]) {
    float v = float(tid);
    b0[tid].a = v;
    b1[tid].a = float2(v);
    b2[tid].a = float3(v);
    b3[tid].a = float4(v);
    b4[tid].a = packed_float3(v);
    b5[tid].a = half(v);
    b6[tid].a = half4(v);
    b7[tid].a = char(tid);
    b8[tid].a = short(tid);
    b9[tid].a = ulong(tid);
    b10[tid].a = tid != 0;
    b11[tid].a = int3(int(tid));
    b12[tid].m = float2x2(v);
    b13[tid].m = float3x3(v);
    b14[tid].m = float4x4(v);
    b15[tid].a[0] = v;
    b16[tid].a[1] = float4(v);
    b17[tid].b = float4(v);
    b18[tid].b = v;
    b19[tid].b = v;
    b20[tid].in.a = float2(v);
    b21[tid].in[1].a = float2(v);
    atomic_store_explicit(b22, tid, memory_order_relaxed);
    atomic_store_explicit(b23, int(tid), memory_order_relaxed);
    atomic_store_explicit(&b24[tid].c, tid, memory_order_relaxed);
    b24[tid].f = v;
}

struct EncTailF { texture2d<float> t; float f; };
struct EncTailF2 { texture2d<float> t; float2 v; };
struct EncTailF4 { texture2d<float> t; float4 v; };
struct EncTailFArr { texture2d<float> t; float a[3]; };
struct EncTailF4Arr { texture2d<float> t; float4 a[2]; };
struct EncTailM22 { texture2d<float> t; float2x2 m; };
struct EncTailC { texture2d<float> t; char c; };
struct EncPlainData { float a; int b; };
struct EncWithPlain { texture2d<float> t; EncPlainData pd; };
struct EncWithPlainArr { texture2d<float> t; EncPlainData pds[2]; };
struct EncIds {
    texture2d<float> t [[id(0)]];
    float f [[id(4)]];
};

kernel void encoders_args(device EncTailF &a0 [[buffer(0)]],
                          device EncTailF2 &a1 [[buffer(1)]],
                          device EncTailF4 &a2 [[buffer(2)]],
                          device EncTailFArr &a3 [[buffer(3)]],
                          device EncTailF4Arr &a4 [[buffer(4)]],
                          device EncTailM22 &a5 [[buffer(5)]],
                          device EncTailC &a6 [[buffer(6)]],
                          device EncWithPlain &a7 [[buffer(7)]],
                          device EncWithPlainArr &a8 [[buffer(8)]],
                          device EncIds &a9 [[buffer(9)]],
                          device float4 *out [[buffer(10)]],
                          uint tid [[thread_position_in_grid]]) {
    uint2 p = uint2(0);
    float4 s = a0.t.read(p) * a0.f;
    s += a1.t.read(p) * a1.v.x;
    s += a2.t.read(p) * a2.v;
    s += a3.t.read(p) * a3.a[2];
    s += a4.t.read(p) * a4.a[1];
    s += a5.t.read(p) * a5.m[1][1];
    s += a6.t.read(p) * float(a6.c);
    s += a7.t.read(p) * a7.pd.a;
    s += a8.t.read(p) * float(a8.pds[1].b);
    s += a9.t.read(p) * a9.f;
    out[tid] = s;
}
