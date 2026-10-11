// A pointer walked through a nested loop, carried across both back edges.
//
// Metal runs this file; the translator reads the sibling `.ll`, which spells the same walk with the
// outer phi seeded from the INNER loop's advanced pointer -- the shape the corpus modules have and
// the one the Metal front end will not emit from any source text, because it strength-reduces the
// outer step to `src + 4`. Both compute the same sum of the same eight elements.
//
// The `.ll` shape is a `StorageBuffer` pointer `OpPhi` advanced by an `OpPtrAccessChain`. It is
// valid SPIR-V, `spirv-val` accepts it, and MoltenVK will not load it: SPIRV-Cross declares the phi
// as a pointer variable and initializes it from the entry access chain with the lvalue instead of
// its address. Run against the pre-fix translator this case comes back `create compute pipeline:
// Initialization of an object has failed`.
//
// The eight inputs are the powers of two from 1 to 128, so every partial sum is an exact integer
// below 2^24 and the total is 255 under any association a fast-math reassociation could choose.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_walk_a_constant_pointer(
    constant float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float acc = 0.0f;
    constant float* p = src;
    for (uint oi = 0; oi < 2u; ++oi) {
        for (uint ii = 0; ii < 4u; ++ii) {
            acc += *p;
            ++p;
        }
    }
    *dst = acc;
}
