// The float-as-signed-int threadgroup atomic min/max idiom -- a `threadgroup float` reduced with
// `atomic_fetch_min/max` on its BITS. It is how every GPU bounding-box builder computes a
// per-threadgroup extent, and until this fixture the corpus had exactly one host for it (an
// 841-line BVH builder), so no authored case reached `native/wg_atomic.rs` at all.
//
// Two lowerings meet here and both are load-bearing:
//
//  * `native/wg_atomic.rs`. Logical SPIR-V forbids an `OpBitcast` of a pointer, so the emitter
//    cannot spell "the int at this float's address". It RETYPES the Workgroup variable's float
//    leaves to the int32 the atomics use -- same tree shape, same byte offsets -- and turns the
//    plain float loads and stores of the same variable into VALUE bitcasts. `Box` is a struct of
//    two float arrays precisely so the retype has to clone a nested tree rather than a scalar.
//  * `passes/workgroup/atomic_loop.rs`. The axis loop is one block, counted, barrier-free and
//    holds two Workgroup atomics, so it is unrolled to its three iterations before the structurizer
//    sees it. `unroll(disable)` keeps the frontend from unrolling it first; without the pragma the
//    AIR arrives already flat and nothing is left to measure.
//
// Why the answer is exact even though eight threads race. Atomic min and max are commutative and
// associative, so the final value of each of the six accumulators does not depend on the
// interleaving. Every input is a small integer-valued float, exactly representable, and the
// comparison happens on the ORDER-PRESERVING int encoding: for a non-negative float the encoding is
// the bit pattern itself, and for a negative one it is the bit pattern with the low 31 bits flipped,
// which reverses the descending order of negative bit patterns into the ascending order the signed
// compare wants. The two axes with negative values are therefore the ones that fail if the flip is
// dropped or if the compare is unsigned rather than signed, and axis 2 is all-positive so the
// unflipped branch is exercised too.
//
// Each accumulator starts from a sentinel the kernel never writes -- `lo` from FLT_MAX (the largest
// encoding any finite float has) and `hi` from -0.0f, whose bit pattern 0x80000000 is the smallest
// signed int. A dropped initial read leaves one of those visible: FLT_MAX or a NaN.
#include <metal_stdlib>
using namespace metal;

struct Box { float lo[3]; float hi[3]; };

// Order-preserving float -> signed int. Involutive, so the same expression decodes.
static inline int order_bits(int i) {
    return (i < 0) ? (i ^ 0x7fffffff) : i;
}

kernel void kernel_threadgroup_float_bbox_atomics(
    device const float* points [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup Box tg;
    if (tid == 0) {
        for (uint a = 0; a < 3; ++a) {
            tg.lo[a] = FLT_MAX;   // bits 0x7f7fffff: >= the encoding of every finite float
            tg.hi[a] = -0.0f;     // bits 0x80000000: the smallest signed int
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float3 p = float3(points[tid * 3 + 0], points[tid * 3 + 1], points[tid * 3 + 2]);
#pragma clang loop unroll(disable)
    for (uint a = 0; a < 3; ++a) {
        int v = order_bits(as_type<int>(p[a]));
        atomic_fetch_min_explicit((threadgroup atomic_int*)&tg.lo[a], v, memory_order_relaxed);
        atomic_fetch_max_explicit((threadgroup atomic_int*)&tg.hi[a], v, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        for (uint a = 0; a < 3; ++a) {
            out[a]     = as_type<float>(order_bits(as_type<int>(tg.lo[a])));
            out[3 + a] = as_type<float>(order_bits(as_type<int>(tg.hi[a])));
        }
    }
}
