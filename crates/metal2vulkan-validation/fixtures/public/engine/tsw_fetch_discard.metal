// Owned fixture: a fragment function without early fragment tests that
// reads a two-channel `half` color attachment (`[[color(1)]]`), discards a fragment whose lane `x`
// there is zero, and writes it back beside a four-channel `half` color 0 it does not read. With
// `w = coverage.x / 2 + coverage.y`, it writes `tint * w` to color 0 and `(w, coverage.y / 4)` to
// color 1. Every value the lanes take from a clear of dyadic fractions is a dyadic fraction a
// `half` holds exactly. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Tested {
    half4 color [[color(0)]];
    half2 coverage [[color(1)]];
};

fragment Tested tsw_fetch_discard(float4 position [[position]],
                                  half2 coverage [[color(1)]],
                                  constant half4 &tint [[buffer(0)]])
{
    if (coverage.x == 0.0h) {
        discard_fragment();
    }
    half w = coverage.x * 0.5h + coverage.y;
    Tested out;
    out.color = tint * w;
    out.coverage = half2(w, coverage.y * 0.25h);
    return out;
}
