// metal-fast-math: off
// sum_regroup_precise: `sum_regroup`'s `regroup` fragment compiled without fast math
// (`lower::arith::reassoc`): no operation carries `reassoc`, so the translator regroups no sum, and
// each keeps the source's grouping
//
//   gx = ((wR + TR) + BR) - (BL + (wL + TL)),   gy = ((wD + BL) + BR) - (TR + (wU + TL)).
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 pos [[position]];
};

vertex VOut sum_regroup_precise_full(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

#define SETUP                                                                                   \
    constexpr sampler S(coord::pixel, filter::nearest, address::clamp_to_edge);                 \
    float ox = cb[0], oy = cb[1], lx = cb[2], ly = cb[3], hx = cb[4], hy = cb[5], st = cb[6];   \
    float2 P = in.pos.xy + float2(ox, oy);                                                      \
    float2 minus = P - st;                                                                      \
    float2 plus = P + st;                                                                       \
    float2 lo = float2(lx, ly);                                                                 \
    float2 hi = float2(hx, hy);                                                                 \
    bool2 inMinus = minus >= lo && minus <= hi;                                                 \
    bool2 inPlus = plus >= lo && plus <= hi;

#define SAMPLE(n, coord) half s##n = t.sample(S, coord).w;

#define SUMS                                                                                    \
    half TLv = all(inMinus) ? sTL : 0.0h;                                                       \
    half BLv = (inMinus.x ? inPlus.y : false) ? sBL : 0.0h;                                     \
    half TRv = (inPlus.x ? inMinus.y : false) ? sTR : 0.0h;                                     \
    half BRv = all(inPlus) ? sBR : 0.0h;                                                        \
    half wR = inPlus.x ? sR * 2.0h : 0.0h;                                                      \
    half wL = inMinus.x ? sL * 2.0h : 0.0h;                                                     \
    half wD = inPlus.y ? sD * 2.0h : 0.0h;                                                      \
    half wU = inMinus.y ? sU * 2.0h : 0.0h;                                                     \
    half gx = ((wR + TRv) + BRv) - (BLv + (wL + TLv));                                          \
    half gy = ((wD + BLv) + BRv) - (TRv + (wU + TLv));                                          \
    return as_type<uint>(sC > 1000.0h ? half2(0.0h) : half2(gx, gy));

fragment uint strict(VOut in [[stage_in]], constant float *cb [[buffer(0)]],
                     texture2d<half> t [[texture(0)]]) {
    SETUP
    SAMPLE(C, P)
    SAMPLE(L, float2(minus.x, P.y))
    SAMPLE(R, float2(plus.x, P.y))
    SAMPLE(U, float2(P.x, minus.y))
    SAMPLE(D, float2(P.x, plus.y))
    SAMPLE(TL, minus)
    SAMPLE(TR, float2(plus.x, minus.y))
    SAMPLE(BL, float2(minus.x, plus.y))
    SAMPLE(BR, plus)
    SUMS
}
