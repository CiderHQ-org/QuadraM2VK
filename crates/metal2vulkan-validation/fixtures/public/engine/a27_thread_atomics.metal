// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: MSL 4.1's atomics on thread memory, which only AIR 2.9
// (deployment target macOS 27) spells, as `air.atomic.private.*`. Each thread runs every
// operation on its own `thread` atomics of `uint`, `int`, `float`, and `bool`, with the
// non-explicit (`seq_cst`, every memory) and explicit orders and memory flags, and records each
// old value and the final ones: nothing another thread does can change them.
//
// Thread t of out (24 words per thread), with x = t * 0x01010101 + 3, s = int(x) - 100, and
// f = float(t) * 0.5: [0] x; [1] x + 5; [2] x + 3; [3] A = (x + 3) & 0xf0f0f0f0; [4] B = A |
// 0x0101; [5] C = B ^ 0xffff; [6] D = min(C, 0x1000) (unsigned); [7] max(D, 0x800); [8] 1 and
// [9] 77 (a compare-exchange that succeeds); [10] 0 and [11] 88 (one that fails and reads 88
// back); [12] 88; [13] s; [14] min(s, -7) (signed); [15] 3; [16] 2; [17] f, [18] f + 1.25,
// [19] f + 0.5, and [20] -2.5 (as bits); [21] x & 1 (a bool); [22] 1; [23] 88. One threadgroup;
// loop-free.
//
// The fixture holds no lifetime marker: LLVM 22 and later read a marker in a one-argument form, so
// a fixture holding one may decode to other type numbers under each `llvm-as` major a gate
// assembles it with. The atomics and the compare-exchanges' `expected` words are
// declared ahead of the switch's only label (on t >> 31, zero for every thread), so every path into
// the switch bypasses their declarations and Clang emits no marker for them (as in
// ob2_sized_array.metal). The header's compare-exchange copies `*expected` into a local of its own,
// which Clang marks once it is inlined. `exchange` below is that function without the copy: it
// calls the same builtin with the same operands, so each `air.atomic.private.cmpxchg.weak` call
// matches the header's in every operand, with `expected` itself in place of the copy. That is the
// only reason it names the header's reserved builtin and the atomic's storage member, `__s`. The
// declarations' order (the header's copies were the first allocas) and each compare-exchange's
// read of its word ahead of the stores keep the module the header form's, instruction for
// instruction, less the markers, so the fixture's translation is the same SPIR-V.
#include <metal_stdlib>
using namespace metal;

template <typename T, typename U, thread_scope S>
METAL_FUNC bool exchange(thread _atomic<T, S> *object, thread T *expected, U desired,
                         memory_order success, memory_order failure, mem_flags flags) {
  return __metal_atomic_compare_exchange_weak_explicit(&object->__s, expected,
                                                       decltype(object->__s)(desired),
                                                       int(success), int(failure), int(S),
                                                       int(flags));
}

kernel void a27_thread_atomics(device uint *out [[buffer(0)]],
                               uint t [[thread_position_in_threadgroup]]) {
  device uint *o = out + 24 * t;
  uint x = t * 0x01010101u + 3u;
  switch (t >> 31) {
    uint stale;
    uint expected;
    thread atomic_uint a;
    thread atomic_int s;
    thread atomic_float f;
    thread atomic_bool b;
  default:
    atomic_store(&a, x);
    o[0] = atomic_fetch_add(&a, 5u);
    o[1] = atomic_fetch_sub_explicit(&a, 2u, memory_order_relaxed);
    o[2] = atomic_fetch_and(&a, 0xf0f0f0f0u);
    o[3] = atomic_fetch_or(&a, 0x0101u);
    o[4] = atomic_fetch_xor(&a, 0xffffu);
    o[5] = atomic_fetch_min(&a, 0x1000u);
    o[6] = atomic_fetch_max(&a, 0x800u);
    o[7] = atomic_exchange(&a, 77u);
    expected = 77u;
    // The non-explicit compare-exchange's arguments: seq_cst over every memory (mem_flags 31).
    bool swapped = exchange(&a, &expected, 88u, memory_order_seq_cst, memory_order_seq_cst,
                            mem_flags::mem_device | mem_flags::mem_threadgroup |
                                mem_flags::mem_texture | mem_flags::mem_object_data |
                                mem_flags::mem_threadgroup_imageblock);
    uint seen = expected;
    o[8] = swapped ? 1u : 0u;
    o[9] = seen;
    stale = 1u;
    swapped = exchange(&a, &stale, 99u, memory_order_acq_rel, memory_order_acquire,
                       mem_flags::mem_device);
    seen = stale;
    o[10] = swapped ? 1u : 0u;
    o[11] = seen;
    o[12] = atomic_load_explicit(&a, memory_order_acquire, mem_flags::mem_none);
    atomic_store_explicit(&s, int(x) - 100, memory_order_release, mem_flags::mem_threadgroup);
    o[13] = as_type<uint>(atomic_fetch_min(&s, -7));
    o[14] = as_type<uint>(atomic_fetch_max(&s, 3));
    o[15] = as_type<uint>(atomic_fetch_add(&s, -1));
    o[16] = as_type<uint>(atomic_load(&s));
    atomic_store(&f, float(t) * 0.5f);
    o[17] = as_type<uint>(atomic_fetch_add(&f, 1.25f));
    o[18] = as_type<uint>(atomic_fetch_sub(&f, 0.75f));
    o[19] = as_type<uint>(atomic_exchange(&f, -2.5f));
    o[20] = as_type<uint>(atomic_load(&f));
    atomic_store(&b, (x & 1u) != 0u);
    o[21] = atomic_exchange(&b, true) ? 1u : 0u;
    o[22] = atomic_load(&b) ? 1u : 0u;
    o[23] = atomic_exchange_explicit(&a, 5u, memory_order_relaxed);
  }
}
