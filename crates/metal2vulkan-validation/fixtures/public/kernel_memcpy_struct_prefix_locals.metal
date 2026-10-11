#include <metal_stdlib>
using namespace metal;

struct Pair {
    float f;
    int i;
};

struct Row {
    float v[4];
    float tail;
};

struct Cfg {
    Pair p;
    float tail;
};

struct Params {
    Row row;
    Cfg cfg;
};

// Semantic twin of the hand-written AIR beside it. The AIR spells two `llvm.memcpy` calls whose
// LENGTH is shorter than the source object -- a prefix copy out of a constant buffer into a local
// -- which is the emitter shape under test. Metal only has to agree on the value, so it reads the
// same members directly.
kernel void memcpy_struct_prefix_locals(
    constant Params& params [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float v;
    if (gid < 4) {
        v = params.row.v[gid];
    } else if (gid == 4) {
        v = params.cfg.p.f;
    } else {
        v = float(params.cfg.p.i);
    }
    out[gid] = v;
}
