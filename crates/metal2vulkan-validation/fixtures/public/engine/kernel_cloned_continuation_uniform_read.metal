// Owned fixture: the compiled replacement for the hand-written top-level
// `kernel_cloned_continuation_uniform_read.ll`, whose `air.struct_type_info` is not the form
// Metal emits (the typed ABI decoder refuses it, `Abi(BadValue)`). This is the same program as
// that fixture's `.metal` twin, so both answer alike. The front end folds the selection into a
// `select`; what this fixture carries is the ABI Metal emits for a `constant` struct-of-structs
// argument (`air.struct_type_info` whose members name the nested struct's own type info) and a
// member read through a nested `getelementptr`. Loop-free.
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

kernel void cloned_continuation_uniform_read(
    constant Params& p [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float v = (gid >= 2 && gid < 4) ? 7.0f : p.b.c2 + 0.5f;
    out[gid] = v;
}
