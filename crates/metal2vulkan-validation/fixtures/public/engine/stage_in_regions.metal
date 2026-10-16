// Owned fixture (written for this repository): kernels whose `[[stage_in]]`
// fetch a stage-in region smaller than the grid clamps. The compute pipeline's stage-input
// descriptor, not the source, chooses each attribute's step (thread position in the grid on x or
// y at a step rate, or through the index buffer), and the encoder's `setStageInRegion:` the
// region: a thread past the region reads the region's last element. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct StageInWord {
    uint v [[attribute(0)]];
};

// A consumer's shape: one 32-bit attribute, each thread writing it plus 0x100 at its position.
kernel void stage_in_region_word(StageInWord in [[stage_in]],
                                 device uint *out [[buffer(0)]],
                                 uint2 g [[thread_position_in_grid]],
                                 uint2 n [[threads_per_grid]])
{
    out[g.y * n.x + g.x] = in.v + 0x100;
}

struct StageInPair {
    uint a [[attribute(0)]];
    uint b [[attribute(1)]];
};

// Two attributes, which a descriptor steps on the two axes or through the index buffer, each
// thread writing them as one word: `a` in the high half and `b` in the low half.
kernel void stage_in_region_pair(StageInPair in [[stage_in]],
                                 device uint *out [[buffer(0)]],
                                 uint2 g [[thread_position_in_grid]],
                                 uint2 n [[threads_per_grid]])
{
    out[g.y * n.x + g.x] = (in.a << 16) | in.b;
}
