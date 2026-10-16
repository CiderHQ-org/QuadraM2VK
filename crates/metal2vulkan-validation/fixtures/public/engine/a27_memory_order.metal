// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: MSL 4.1's memory orders, memory flags, and barrier memory
// scopes, which only AIR 2.9 (deployment target macOS 27) spells. Every memory atomic carries its
// `mem_flags` and every barrier its `memory_order` and memory scope:
//
//  * a non-explicit atomic (`atomic_store`, `atomic_load`, `atomic_fetch_add`) is `seq_cst` over
//    every memory (`mem_flags` 31, `mem_object_data` included);
//  * an `_explicit` atomic takes an order and the memories it applies to: an acquire load of
//    threadgroup memory, a release add on device memory, a relaxed maximum and an `acq_rel` OR
//    naming texture memory, an `acq_rel`/`acquire` compare-exchange naming device memory;
//  * a device atomic of `thread_scope_threadgroup` passes scope 1;
//  * barriers take `relaxed`, `acquire`, `release`, `acq_rel`, and `seq_cst`, and a memory scope
//    of their own: `thread_scope_simdgroup` and `thread_scope_device` on a threadgroup barrier,
//    `thread_scope_thread` (an execution barrier alone), `thread_scope_threadgroup`.
//
// a27_memory_order: one threadgroup of 32 threads over Block. Every value it writes is independent
// of how the threads interleave: the threadgroup sum is read after an `acq_rel` and a `seq_cst`
// barrier; each exchange reads a word its partner wrote before a barrier that orders that memory
// among both (t ^ 1 is in t's simdgroup at any even width); the device counters are sums, a
// maximum, and an OR that nothing reads during the dispatch; and each compare-exchange works on a
// slot only its own thread touches. The release, acquire, relaxed, and thread-scope barriers order
// nothing the kernel depends on. With counters[16 + t] = 7 and every other word zero before the
// dispatch, after it: counters[0] = 32, [1] = 496, [2] = 31, [3] = 32, [4] = 0xffffffff,
// [5..16) = 0, [16 + t] = 100 + t; out[t] = 1520 (the sum of 3u + 1 over u < 32), out[32 + t]
// = 7 (the slot's value before its exchange), out[64 + t] = 1 (the exchange succeeded),
// out[96 + t] = 7t + 3, out[128 + t] = 7 (t ^ 31) + 3, out[160 + t] = 5 (t ^ 1) + 2. Loop-free.
//
// a27_wide_and_float: a relaxed 64-bit maximum naming device memory and a non-explicit float add,
// apart (they need 64-bit and float atomics). Over n threads from zero: wide = (n - 1) << 33, sum
// = n (n - 1) / 2 (exact while below 2^24). Loop-free.
//
// a27_fragment_atomics: a fragment function's non-explicit atomics, `seq_cst` over every memory,
// of which a render stage has device and texture memory only. One invocation (a 1x1 pass) from
// count = {a, b, c}: count = {a + 1, b, a + b}. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Block {
  atomic_uint counters[48];
  uint out[192];
};

kernel void a27_memory_order(device Block &b [[buffer(0)]],
                             uint t [[thread_position_in_threadgroup]]) {
  threadgroup atomic_uint sum;
  threadgroup uint words[32];
  if (t == 0) {
    atomic_store(&sum, 0u);
  }
  words[t] = 3u * t + 1u;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  atomic_fetch_add(&sum, words[t ^ 1u]);
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
  uint total = atomic_load_explicit(&sum, memory_order_acquire, mem_flags::mem_threadgroup);
  words[t] = 5u * t + 2u;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_simdgroup);
  uint pair = words[t ^ 1u];
  b.out[96 + t] = 7u * t + 3u;
  threadgroup_barrier(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  uint across = b.out[96 + (t ^ 31u)];
  atomic_fetch_add_explicit(&b.counters[0], 1u, memory_order_release, mem_flags::mem_device);
  atomic_fetch_add(&b.counters[1], t);
  atomic_fetch_max_explicit(&b.counters[2], t, memory_order_relaxed, mem_flags::mem_texture);
  device atomic<uint, thread_scope_threadgroup> *scoped =
      reinterpret_cast<device atomic<uint, thread_scope_threadgroup> *>(&b.counters[3]);
  atomic_fetch_add_explicit(scoped, 1u, memory_order_relaxed);
  atomic_fetch_or_explicit(&b.counters[4], 1u << t, memory_order_acq_rel, mem_flags::mem_texture);
  uint expected = 7u;
  bool exchanged = atomic_compare_exchange_weak_explicit(
      &b.counters[16 + t], &expected, 100u + t, memory_order_acq_rel, memory_order_acquire,
      mem_flags::mem_device);
  threadgroup_barrier(mem_flags::mem_device, memory_order_release);
  threadgroup_barrier(mem_flags::mem_device, memory_order_acquire);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_device, memory_order_seq_cst, thread_scope_thread);
  simdgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  b.out[t] = total;
  b.out[32 + t] = expected;
  b.out[64 + t] = exchanged ? 1u : 0u;
  b.out[128 + t] = across;
  b.out[160 + t] = pair;
}

struct Wide {
  atomic_ulong wide;
  atomic_float sum;
};

kernel void a27_wide_and_float(device Wide &w [[buffer(0)]],
                               uint t [[thread_position_in_grid]]) {
  atomic_max_explicit(&w.wide, ulong(t) << 33, memory_order_relaxed, mem_flags::mem_device);
  atomic_fetch_add(&w.sum, float(t));
}

fragment void a27_fragment_atomics(device atomic_uint *count [[buffer(0)]]) {
  uint n = atomic_fetch_add(&count[0], 1u);
  uint m = atomic_load(&count[1]);
  atomic_store(&count[2], n + m);
}
