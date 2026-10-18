// sum_regroup: Metal's regrouping of a fast-math sum (`lower::arith::reassoc`). A
// gradient filter's seed pass, written from a structural description of its shape (no guest code
// or constants): each fragment reads seven constant-buffer floats (an offset, the bounds and a
// step), samples one channel of the nine nearest texels around its pixel, keeps a neighbour only
// where it lies inside the bounds, weighs the four edges by two, and sums
//
//   gx = ((wR + TR) + BR) - (BL + (wL + TL)),   gy = ((wD + BL) + BR) - (TR + (wU + TL))
//
// compiled with the default fast math. The frontend regroups each sum itself, and Metal's backend
// regroups the AIR's sum again by its ranks, so the AIR's grouping is not the one Metal computes.
// Each fragment returns its two sums' bits, or zeros where the centre sample is above 1000 (none
// is), which keeps the centre sample live. Measured on an Apple M4:
// - regroup: samples in the order C L R U D TL TR BL BR. Metal computes
//   gx = (((wR - (wL + TL)) + TR) - BL) + BR and gy = ((wD - ((wU + TL) + TR)) + BL) + BR.
// - tied: samples in the order TL U BL C L BR R TR D, where leaves tie in rank and keep the tree's
//   order, which is also their latest reads' order. Metal computes
//   gx = (((-TL - (BL + wL)) + BR) + wR) + TR and gy = (((BL - (wU + TL)) + BR) - TR) + wD.
// - unpinned: `regroup` whose top-left leaf is the centre sample where the top-left one is
//   negative: a selection of a selection, which ranks it above the later top-right sample's leaf,
//   so the rank order is not the order of the leaves' reads and does not pin the grouping. Each
//   sum keeps the AIR's grouping and notes the rewrites Metal may make.
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 pos [[position]];
};

vertex VOut sum_regroup_full(uint vid [[vertex_id]]) {
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

#define SUMS SUMS_OF(sTL)
#define SUMS_OF(tl)                                                                             \
    half TLv = all(inMinus) ? tl : 0.0h;                                                        \
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

fragment uint regroup(VOut in [[stage_in]], constant float *cb [[buffer(0)]],
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

fragment uint tied(VOut in [[stage_in]], constant float *cb [[buffer(0)]],
                   texture2d<half> t [[texture(0)]]) {
    SETUP
    SAMPLE(TL, minus)
    SAMPLE(U, float2(P.x, minus.y))
    SAMPLE(BL, float2(minus.x, plus.y))
    SAMPLE(C, P)
    SAMPLE(L, float2(minus.x, P.y))
    SAMPLE(BR, plus)
    SAMPLE(R, float2(plus.x, P.y))
    SAMPLE(TR, float2(plus.x, minus.y))
    SAMPLE(D, float2(P.x, plus.y))
    SUMS
}

fragment uint unpinned(VOut in [[stage_in]], constant float *cb [[buffer(0)]],
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
    SUMS_OF(sTL < 0.0h ? sC : sTL)
}
