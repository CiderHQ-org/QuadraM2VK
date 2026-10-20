// hwd_half_reads: every `half` bit pattern read back and used, one thread per pattern (256
// threadgroups of 256 threads). Each thread stores its pattern `src[g]` to threadgroup words held
// as `half`s, then reads back its mirror's (`y`, the cell 255 - t) and uses it only in a sum, a
// widening to `float` and a compare; reads its neighbour's (`m`, the cell t + 1 mod 256) and moves
// it through a select to memory; and reads the pattern `g ^ 1` from the buffer (`b`) into a
// product. The operands come from buffers so no fold removes the operations: `k` holds -0, 1 and 1,
// and `flag` is 0. Metal's answers: `sums[g]` is `y` (a NaN `0x7E00`), `wides[g]` the `float` of
// `y` (a NaN `0x7FC00000`), `less[g]` whether `y < 1`, `moved[g]` the bits of `m`, and `prods[g]`
// `b` (a NaN `0x7E00`).
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void hwd_half_reads(device const half *src [[buffer(0)]],
                           device half *sums [[buffer(1)]],
                           device float *wides [[buffer(2)]],
                           device uint *less [[buffer(3)]],
                           device half *moved [[buffer(4)]],
                           device half *prods [[buffer(5)]],
                           constant half *k [[buffer(6)]],
                           constant uint &flag [[buffer(7)]],
                           uint g [[thread_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint words[128];
    threadgroup half *cells = (threadgroup half *)words;
    cells[t] = src[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half y = cells[255 - t];
    sums[g] = y + k[0];
    wides[g] = float(y);
    less[g] = y < k[2] ? 1u : 0u;
    half m = cells[(t + 1) & 255];
    moved[g] = flag == 0 ? m : k[1];
    half b = src[g ^ 1];
    prods[g] = b * k[1];
}
