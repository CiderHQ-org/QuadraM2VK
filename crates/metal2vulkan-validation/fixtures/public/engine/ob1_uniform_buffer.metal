// Owned fixture: loads past a `constant` buffer of a known size (a 16-byte
// struct, so its window is bounded and robust descriptors view it as a uniform buffer), the
// uniform buffer containment form. Eight threads; thread t loads word idx[t] of blk into out[t].
// The case's idx puts loads 0 to 3 in blk and loads 4 to 7 past its end (words 4, 7, 20, 63: up to
// byte 256). Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Block {
    uint4 v;
};

kernel void ob1_uniform_buffer(device const uint *idx [[buffer(0)]],
                               constant Block &blk [[buffer(1)]],
                               device uint *out [[buffer(2)]],
                               uint t [[thread_position_in_grid]]) {
    constant uint *words = reinterpret_cast<constant uint *>(&blk);
    out[t] = words[idx[t]];
}
