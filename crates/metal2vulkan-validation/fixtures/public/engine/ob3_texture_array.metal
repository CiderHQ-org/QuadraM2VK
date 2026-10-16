// Owned fixture: reads through an index past a top-level texture array, the
// handle array index containment form. The array `tex` takes texture slots 0 and 1, and the array
// `spare` slots 2 and 3, so an index of 2 or 3 into `tex` names a slot the case binds (one of
// `spare`'s). Eight threads; thread t reads texel (0, 0) of tex[idx[t]] into out[t], and texel
// (0, 0) of spare[t & 1] into out[8 + t]. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void ob3_texture_array(array<texture2d<uint, access::read>, 2> tex [[texture(0)]],
                              array<texture2d<uint, access::read>, 2> spare [[texture(2)]],
                              device const uint *idx [[buffer(0)]],
                              device uint *out [[buffer(1)]],
                              uint t [[thread_position_in_grid]]) {
    out[t] = tex[idx[t]].read(uint2(0, 0)).x;
    out[8 + t] = spare[t & 1].read(uint2(0, 0)).x;
}
