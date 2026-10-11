// A `[[threadgroup]]` buffer whose element is an 80-byte AIR struct, bound by the caller at 2560
// bytes -- 32 elements.
//
// The translator used to compute that element count against a type-definition map collected before
// it synthesized the element type, and `spirv_size_align` answers one word for an id it cannot look
// up. So this element measured 4 bytes, the 2560-byte binding named 640 elements, and the module
// declared 51200 bytes of threadgroup memory. Metal refuses any pipeline declaring more than 32768,
// so before the fix this case could not run at all.
//
// Nothing in the struct is one word, on purpose: `uint4 v[5]` is 80 bytes with no padding to argue
// about, and an element size read as anything but 80 gives a different array.
//
// Every slot that is read is written by some thread before the barrier, so no thread observes
// Metal's undefined initial threadgroup contents, and the answer does not depend on the executor's
// zero-fill. Each lane reads two OTHER lanes' slots, which is what makes the stride matter: if
// elements were sized wrong the neighbours would not be where the writer put them.
#include <metal_stdlib>
using namespace metal;

struct Wide { uint4 v[5]; };

kernel void kernel_threadgroup_struct_scratch(
    device uint* out [[buffer(0)]],
    threadgroup Wide* scratch [[threadgroup(0)]],
    uint lid [[thread_position_in_threadgroup]])
{
    scratch[lid].v[0] = uint4(lid * 3u + 1u, lid * 5u + 2u, lid * 7u + 3u, lid * 11u + 4u);
    scratch[lid].v[4] = uint4(lid * 13u + 5u, 0u, 0u, 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint a = (lid + 1u) & 31u;
    uint b = (lid + 2u) & 31u;
    // 16a + 11b + 10, with a and b two different neighbours of this lane.
    out[lid] = scratch[a].v[0].x + scratch[b].v[0].w + scratch[a].v[4].x;
}
