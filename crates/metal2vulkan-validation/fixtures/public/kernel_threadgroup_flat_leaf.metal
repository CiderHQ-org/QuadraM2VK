#include <metal_stdlib>
using namespace metal;

// Four identical scalar leaves, so the flattened index carries both the record and the member.
struct Bucket {
    uint a;
    uint b;
    uint c;
    uint d;
};

// The threadgroup object is a KERNEL PARAMETER, so the translator sees a Workgroup root whose
// pointee is an array of records while every load and store through it wants one `uint` leaf.
kernel void kernel_threadgroup_flat_leaf(threadgroup Bucket *tile [[threadgroup(0)]],
                                         device uint *out [[buffer(0)]],
                                         device const uint *in [[buffer(1)]],
                                         uint tid [[thread_position_in_threadgroup]]) {
    uint v = in[tid];
    tile[tid].a = v;
    tile[tid].b = v + 1u;
    tile[tid].c = v + 2u;
    tile[tid].d = v + 3u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A whole-record assignment. The frontend emits it as `llvm.memcpy` between two record
    // pointers with no member index, which is the shape that leaves a flat one-index access
    // chain for the leaf repair to decompose.
    uint src = (tid + 1u) & 7u;
    tile[tid + 8u] = tile[src];
    threadgroup Bucket *p = &tile[tid + 8u];
    out[tid] = p->a + p->b * 2u + p->c * 4u + p->d * 8u;
}
