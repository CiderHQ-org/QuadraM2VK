// Owned fixture: a quad operation in a branch's arm of a shape no probe of Metal's speculation
// covers. Metal's backend runs a branch's short arms of plain computation in every lane of the
// branch's block, so a quad broadcast there answers over the block's lanes; the translation
// speculates the arm shapes measured on Apple's GPU as Metal does (`air::canon::speculate`). An arm
// holding a math function other than `sqrt` is not among them.
// - `quad_first_in_an_unmeasured_arm`: `quad_broadcast_first` beside `sin` in the arm. The branch
//   keeps the AIR's lanes, and the pipeline records `UnmeasuredGroupSpeculation`.
// - `sin_in_an_unmeasured_arm`: the same arm without the broadcast: no group operation, no note.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void quad_first_in_an_unmeasured_arm(device uint *out [[buffer(1)]],
                                            uint t [[thread_position_in_grid]])
{
    uint r = 7777u;
    if (t % 3 != 0) {
        r = quad_broadcast_first(t * 10u + uint(sin(float(t)) * 4.0f + 4.0f));
    }
    out[t] = r;
}

kernel void sin_in_an_unmeasured_arm(device uint *out [[buffer(1)]],
                                     uint t [[thread_position_in_grid]])
{
    uint r = 7777u;
    if (t % 3 != 0) {
        r = t * 10u + uint(sin(float(t)) * 4.0f + 4.0f);
    }
    out[t] = r;
}
