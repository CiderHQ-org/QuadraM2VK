// Owned fixture: a fragment function without early fragment tests
// that reads its color attachment (`[[color(0)]]`), writes back half of it plus a quarter, and
// returns the depth its buffer holds (`[[depth(any)]]`). Beside a depth attachment its color store
// in the interlocked storage form precedes the depth test, which then tests the depth it returns.
// The buffer lets a draw return any depth bits: a NaN, a -0, a value past 1 or below 0. Every
// color value it takes from a clear of dyadic fractions is a dyadic fraction a `float` holds
// exactly. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct FetchedDepth {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

fragment FetchedDepth fetch_returned_depth(float4 color [[color(0)]],
                                           constant float &depth [[buffer(0)]])
{
    FetchedDepth out;
    out.color = color * 0.5f + 0.25f;
    out.depth = depth;
    return out;
}
