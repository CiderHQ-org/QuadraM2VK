#include <metal_stdlib>
using namespace metal;

struct Box {
    float2 v;
    float f1;
    float f2;
};

// Semantic twin of the hand-written AIR beside it. The AIR spells three `llvm.memset` calls whose
// LENGTH does not equal the object the destination pointer names: two clear a PREFIX of a local
// (4 of 16 bytes and 12 of 16), and the third clears sixteen bytes through a pointer to a single
// device float. Metal only has to agree on the value, so it assigns the same members directly.
kernel void zero_a_struct_prefix(
    constant Box& src [[buffer(0)]],
    device float* out [[buffer(1)]])
{
    Box a = src;
    Box b = src;
    a.v.x = 0.0f;
    b.v = float2(0.0f);
    b.f1 = 0.0f;
    out[0] = a.v.x;
    out[1] = a.v.y;
    out[2] = a.f1;
    out[3] = a.f2;
    out[4] = b.v.x;
    out[5] = b.v.y;
    out[6] = b.f1;
    out[7] = b.f2;
    out[8] = 0.0f;
    out[9] = 0.0f;
    out[10] = 0.0f;
    out[11] = 0.0f;
}
