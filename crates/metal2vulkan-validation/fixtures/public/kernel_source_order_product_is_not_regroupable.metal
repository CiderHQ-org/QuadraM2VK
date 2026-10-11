// A three-term product whose two association orders round to different floats.
//
// Metal decides its float relaxations from the module's math mode and the expression's
// `precise`-ness; SPIR-V has no equivalent module state, so a permission LLVM withheld has to
// travel on the emitted instruction. The emitter used to say only `NoContraction`, which forbids
// fusing a multiply into an add and leaves REASSOCIATION permitted -- so an emitted module stated
// less than its AIR did, and MoltenVK, whose own fast-math default is on, was free to regroup.
//
// `a` and `b` are both `1 + 2^-23`, the smallest float above one, and `c` is 3. In source order
// `a*b` is `1 + 2^-22 + 2^-46`, which rounds to exactly `1 + 2^-22`, and three times that is
// `3 + 2^-22 + 2^-21` -- exact, because the ulp at 3 is 2^-22. Regrouped, `b*c` is `3 + 3*2^-23`,
// which is 1.5 ulp above 3 and rounds to even at `3 + 2^-21`, and multiplying that by `a` gives
// `3 + 2^-21 + 2^-22`. The two orders are one ulp apart: 3.0000007152557373 (0x40400003) in source
// order against 3.0000009536743164 (0x40400004) regrouped, so the single output word names which
// grouping ran. Nothing else can move it -- there is no add to fuse into, no division to
// reciprocate, and every input is exact -- so the answer is the association order and nothing else.
//
// The `.ll` spells both multiplies with an empty flag run, which is what `-fno-fast-math` produces
// and what the Metal oracle honours when it builds the metallib from the AIR.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_source_order_product_is_not_regroupable(
    device float* out [[buffer(0)]],
    device const float* in [[buffer(1)]])
{
    float a = in[0];
    float b = in[1];
    float c = in[2];
    out[0] = (a * b) * c;
}
