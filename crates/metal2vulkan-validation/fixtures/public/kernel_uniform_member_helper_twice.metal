#include <metal_stdlib>
using namespace metal;

struct Coef {
    float c0;
    float c1;
    float c2;
    float c3;
};

struct Params {
    Coef a;
    Coef b;
    float scale;
};

// Multi-block, so the AIR-text inliner leaves it as a called function and its `constant Coef&`
// parameter has to reach it as a cursor on the uniform buffer's descriptor root.
static __attribute__((noinline)) float poly(constant Coef& c, float x)
{
    if (x < 0.0f) {
        return c.c0;
    }
    return ((c.c3 * x + c.c2) * x + c.c1) * x + c.c0;
}

kernel void uniform_member_helper_twice(
    constant Params& p [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float r = poly(p.a, float(gid));
    if (gid & 1u) {
        r = r + poly(p.a, float(gid) + 1.0f);
    }
    out[gid] = r * p.scale;
}
