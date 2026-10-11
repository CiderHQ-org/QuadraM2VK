#include <metal_stdlib>
using namespace metal;

// Two DISTINCT device buffers whose declared element widths differ -- a byte buffer viewed as
// words, and a word buffer -- merged and then read through the merge. Logical SPIR-V has no
// pointer value that can be either descriptor, so the merge has to be replayed per arm in the
// VALUE domain: one load from each buffer and a select of the results.
kernel void merge_buffers_of_two_element_widths(device const uchar *bytes [[buffer(0)]],
                                                device const uint *words [[buffer(1)]],
                                                device uint *out [[buffer(2)]],
                                                constant uint &pick [[buffer(3)]],
                                                uint gid [[thread_position_in_grid]]) {
    device const uint *selected = ((gid & 1u) == pick) ? (device const uint *)bytes : words;
    out[gid * 2u + 0u] = selected[gid];
    out[gid * 2u + 1u] = selected[gid ^ 7u];
}
