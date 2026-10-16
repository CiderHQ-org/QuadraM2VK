// Owned fixture: quad_shuffle in a compute kernel.
// Metal compute quads are four consecutive linear threads of the threadgroup, so with
// t = thread_index_in_threadgroup the expected value is
//     out[t] = (t & ~3u) | ((t + 1u) & 3u)
// for every thread of a full quad. Vulkan does not tie subgroup lanes to LocalInvocationIndex,
// so a lowering must derive the quad from the linear index or refuse (Unsupported{SimdWidth}).
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void quad_shuffle_rotate(device uint *out [[buffer(0)]],
                                uint t [[thread_index_in_threadgroup]]) {
    out[t] = quad_shuffle(t, (t + 1u) & 3u);
}
