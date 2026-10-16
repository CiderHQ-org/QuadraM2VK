// Owned fixture: kernel stage-input fetches past their buffer, the stage
// input containment form. Thread t writes the attribute it fetched (element t of the stage-input
// buffer) to out[t]. The case's stage-input buffer holds four elements and the grid has eight
// threads, so threads 4 to 7 fetch past it. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct In {
    uint v [[attribute(0)]];
};

kernel void ob3_stage_input(In in [[stage_in]],
                            device uint *out [[buffer(0)]],
                            uint t [[thread_position_in_grid]]) {
    out[t] = in.v;
}
