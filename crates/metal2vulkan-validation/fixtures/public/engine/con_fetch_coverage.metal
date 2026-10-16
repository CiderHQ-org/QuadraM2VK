// Owned fixture: a fragment function with early fragment tests that reads a
// two-channel `half` color attachment (`[[color(1)]]`) and writes it back beside a four-channel
// color it does not read. With `w = coverage.x / 2 + coverage.y`, it writes `tint * w` to color 0
// and `(w, coverage.y / 4)` to color 1, so `n` fragments over one pixel in order leave a value in
// each lane of color 1 that a fragment reading a stale attachment would not. Every value the lanes
// take from a clear of dyadic fractions is a dyadic fraction a `half` holds exactly. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Coverage {
    half4 color [[color(0)]];
    half2 coverage [[color(1)]];
};

[[early_fragment_tests]]
fragment Coverage con_fetch_coverage(float4 position [[position]],
                                     half2 coverage [[color(1)]],
                                     constant half4 &tint [[buffer(0)]])
{
    half w = coverage.x * 0.5h + coverage.y;
    Coverage out;
    out.color = tint * w;
    out.coverage = half2(w, coverage.y * 0.25h);
    return out;
}
