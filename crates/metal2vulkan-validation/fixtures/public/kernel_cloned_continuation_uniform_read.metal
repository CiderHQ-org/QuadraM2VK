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
};

// Semantic twin of the hand-written AIR beside it. The AIR spells the same answer through a
// nested selection whose in-arm continuation the enclosing arm also reaches, which is the CFG the
// structurizer has to privatize by CLONING -- the shape under test. Metal only has to agree on the
// value.
kernel void cloned_continuation_uniform_read(
    constant Params& p [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float v = (gid >= 2 && gid < 4) ? 7.0f : p.b.c2 + 0.5f;
    out[gid] = v;
}
