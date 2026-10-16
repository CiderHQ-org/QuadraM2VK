// Owned fixture: argument buffers each holding one
// two-element array of structs, so a pipeline's reflection states which struct arrays it gives
// stride 0 (an array of resource-only structs) and which their element size (an array of
// plain-data structs): structs mixing a texture, sampler or pointer with a constant in both
// orders, a resource-only struct, a plain-data struct, structs nesting each, and a constant array
// as a control. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ArrTexF { texture2d<float> t; float f; };
struct ArrFTex { float f; texture2d<float> t; };
struct ArrSmpF { sampler s; float f; };
struct ArrPtrF { device float *p; float f; };
struct ArrTex { texture2d<float> t; };
struct ArrPD { float a; int b; };
struct ArrNestPD { ArrPD pd; float c; };
struct ArrNestTex { ArrTex in; float c; };
struct ArrF4Tex { float4 v; texture2d<float> t; };

struct AbTexF { ArrTexF a[2]; };
struct AbFTex { ArrFTex a[2]; };
struct AbSmpF { ArrSmpF a[2]; };
struct AbPtrF { ArrPtrF a[2]; };
struct AbTex { ArrTex a[2]; };
struct AbPD { ArrPD a[2]; };
struct AbNestPD { ArrNestPD a[2]; };
struct AbNestTex { ArrNestTex a[2]; };
struct AbF4Tex { ArrF4Tex a[2]; };
struct AbF { float a[3]; };

kernel void encoder_arrays(device AbTexF &a0 [[buffer(0)]],
                           device AbFTex &a1 [[buffer(1)]],
                           device AbSmpF &a2 [[buffer(2)]],
                           device AbPtrF &a3 [[buffer(3)]],
                           device AbTex &a4 [[buffer(4)]],
                           device AbPD &a5 [[buffer(5)]],
                           device AbNestPD &a6 [[buffer(6)]],
                           device AbNestTex &a7 [[buffer(7)]],
                           device AbF4Tex &a8 [[buffer(8)]],
                           device AbF &a9 [[buffer(9)]],
                           device float4 *out [[buffer(10)]],
                           uint tid [[thread_position_in_grid]]) {
    uint2 p = uint2(0);
    float2 c = float2(0.5);
    float4 s = a0.a[1].t.read(p) * a0.a[1].f;
    s += a1.a[1].t.read(p) * a1.a[1].f;
    s += a4.a[1].t.sample(a2.a[1].s, c) * a2.a[1].f;
    s += float4(a3.a[1].p[0] * a3.a[1].f);
    s += a4.a[1].t.read(p);
    s += float4(a5.a[1].a + float(a5.a[1].b));
    s += float4(a6.a[1].pd.a + a6.a[1].c);
    s += a7.a[1].in.t.read(p) * a7.a[1].c;
    s += a8.a[1].t.read(p) * a8.a[1].v;
    s += float4(a9.a[2]);
    out[tid] = s;
}
