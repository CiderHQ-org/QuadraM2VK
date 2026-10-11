// The four atomic operations the corpus reaches but no authored case did: a device XOR, a signed
// device subtract, a FLOAT device subtract, and a threadgroup unsigned subtract.
//
// The float subtract is the one with something to prove. SPIR-V has no atomic float subtract, so
// `air.atomic.global.sub.f32` is lowered as `OpFNegate` followed by `OpAtomicFAddEXT` -- an
// identity in IEEE arithmetic, but an identity nothing had measured. The other three are covered
// here because they cost one line each and the corpus has one or two sources apiece.
//
// Every accumulator is order-independent, which is what makes an exact comparison of a 32-thread
// race meaningful at all:
//
//  * XOR is commutative and associative over the 32 distinct per-thread words, and the words are
//    `tid * 2654435761` truncated to 32 bits, so they overlap in every bit position -- a
//    disjoint-bit pattern would have made XOR, OR and ADD agree.
//  * The two integer subtractions total `1 + 2 + ... + 32 = 528` however they interleave. The
//    threadgroup one starts at zero and therefore UNDERFLOWS to `2^32 - 528`, which is defined and
//    is a different number from any signed reading of it.
//  * The float subtraction starts at 4096 and every partial value is an integer in [3568, 4096].
//    Binary32 represents every integer below 2^24 exactly, so each step is exact and the total is
//    3568 whatever order the 32 threads run in -- the result does not depend on association, which
//    an exact float comparison of a race otherwise could not survive.
//
// Each accumulator also starts from a value the kernel never writes, so an implementation that
// dropped the initial read answers differently in all four.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_atomic_xor_and_subtract(
    device atomic_uint* unsigned_words [[buffer(0)]],
    device atomic_int* signed_total [[buffer(1)]],
    device atomic_float* float_total [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup atomic_uint tile_total;
    if (tid == 0) {
        atomic_store_explicit(&tile_total, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    atomic_fetch_xor_explicit(&unsigned_words[0], tid * 2654435761u, memory_order_relaxed);
    atomic_fetch_sub_explicit(signed_total, int(tid) + 1, memory_order_relaxed);
    atomic_fetch_sub_explicit(float_total, float(tid + 1u), memory_order_relaxed);
    atomic_fetch_sub_explicit(&tile_total, tid + 1u, memory_order_relaxed);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        atomic_store_explicit(&unsigned_words[1],
                              atomic_load_explicit(&tile_total, memory_order_relaxed),
                              memory_order_relaxed);
    }
}
