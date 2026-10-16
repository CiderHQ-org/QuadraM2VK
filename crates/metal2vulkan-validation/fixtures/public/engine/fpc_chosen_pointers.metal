// Owned MSL: kernels that call through function pointers
// chosen at run time among functions' addresses, not a table's entries alone. The visible
// functions come from `fpc_pointer_functions.metal`, which the pipeline links; the `own_*`
// functions are this module's and not visible. Each kernel writes word `tid` of buffer 0. Every
// loop runs four times; every recursion ends at its input's depth.
#include <metal_stdlib>
using namespace metal;

using op_t = uint(uint);
using fp_t = op_t *;

[[visible]] uint fpc_inc(uint x);
[[visible]] uint fpc_dbl(uint x);
[[visible]] uint fpc_sq(uint x);
[[visible]] uint fpc_rec_even(uint n);
[[visible]] uint fpc_rec_odd(uint n);

static uint own_inc(uint x) { return x + 1u; }
static uint own_dbl(uint x) { return x * 2u; }
static uint own_sq(uint x) { return x * x; }

// A select of two addresses: `fpc_inc` for odd threads, `fpc_dbl` for even ones.
kernel void fpc_select(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    op_t *f = (tid & 1u) != 0u ? fpc_inc : fpc_dbl;
    out[tid] = f(tid);
}

// A phi of three addresses, `tid % 3` choosing; each branch writes buffer 1, so they stay
// branches.
kernel void fpc_phi(device uint *out [[buffer(0)]], device uint *side [[buffer(1)]],
                    uint tid [[thread_position_in_grid]]) {
    op_t *f;
    uint r = tid % 3u;
    if (r == 0u) {
        side[tid] = 10u;
        f = fpc_inc;
    } else if (r == 1u) {
        side[tid] = 11u;
        f = fpc_dbl;
    } else {
        side[tid] = 12u;
        f = fpc_sq;
    }
    out[tid] = f(tid + 5u);
}

// A pointer carried around a loop: each step calls it, then chooses `fpc_inc` for an odd result
// and `fpc_dbl` for an even one.
kernel void fpc_loop(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    op_t *f = fpc_inc;
    uint acc = tid;
    for (uint i = 0u; i < 4u; ++i) {
        acc = f(acc);
        if ((acc & 1u) != 0u) {
            f = fpc_inc;
        } else {
            f = fpc_dbl;
        }
    }
    out[tid] = acc;
}

// Stored to a private array, loaded back at index `tid % 3`.
kernel void fpc_private_array(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    op_t *fs[3] = {fpc_inc, fpc_dbl, fpc_sq};
    out[tid] = fs[tid % 3u](tid);
}

// Compared for equality: two pointers loaded at run-time indices, a function's address, null.
// Bit 0: the two are equal; bit 1: the first is `fpc_inc`'s; bit 2: it is not null; bit 3: the
// second is `fpc_sq`'s.
kernel void fpc_compare(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    op_t *fs[4] = {fpc_inc, fpc_dbl, fpc_sq, fpc_inc};
    op_t *inc_p = fpc_inc;
    op_t *sq_p = fpc_sq;
    op_t *f = fs[tid % 4u];
    op_t *g = fs[(tid / 4u) % 4u];
    uint r = 0u;
    r |= f == g ? 1u : 0u;
    r |= f == inc_p ? 2u : 0u;
    r |= f != nullptr ? 4u : 0u;
    r |= g == sq_p ? 8u : 0u;
    out[tid] = r;
}

// Stored to threadgroup memory by four threads (`fpc_inc` at odd slots, `fpc_sq` at even ones),
// loaded by every thread at slot `tid & 3` after a barrier.
kernel void fpc_threadgroup(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    threadgroup fp_t slot[4];
    if (lid < 4u) {
        if ((lid & 1u) != 0u) {
            slot[lid] = fpc_inc;
        } else {
            slot[lid] = fpc_sq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[tid] = slot[tid & 3u](tid);
}

// Converted to a 64-bit integer and written to buffer 1, the neighbour's read back after a
// barrier, converted to a pointer and called.
kernel void fpc_int_back(device uint *out [[buffer(0)]], device ulong *bits [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    op_t *f = (tid & 1u) != 0u ? fpc_inc : fpc_dbl;
    bits[tid] = reinterpret_cast<ulong>(f);
    threadgroup_barrier(mem_flags::mem_device);
    op_t *g = reinterpret_cast<op_t *>(bits[tid ^ 1u]);
    out[tid] = g(tid);
}

// A table's entry `tid % 3` merged with `fpc_sq`'s address (the entry for threads with bit 2),
// called, and each compared with an address: plus 1000 when the entry is `fpc_inc`'s, plus 2000
// when the merged pointer is `fpc_sq`'s.
kernel void fpc_table_mix(device uint *out [[buffer(0)]],
                          visible_function_table<op_t> t [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    op_t *e = t[tid % 3u];
    op_t *s = fpc_sq;
    op_t *i = fpc_inc;
    op_t *f = (tid & 4u) != 0u ? e : s;
    uint r = f(tid);
    r += e == i ? 1000u : 0u;
    r += f == s ? 2000u : 0u;
    out[tid] = r;
}

// A table's entry `tid % 3` and `fpc_sq`'s address stored to a private array, the entry called
// by threads whose bit 1 is clear and `fpc_sq` by the others.
kernel void fpc_table_store(device uint *out [[buffer(0)]],
                            visible_function_table<op_t> t [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
    op_t *s = fpc_sq;
    op_t *fs[2] = {t[tid % 3u], s};
    out[tid] = fs[(tid >> 1u) & 1u](tid);
}

// Recursion through pointers chosen at run time: `fpc_rec_odd` for odd threads, `fpc_rec_even`
// for even ones, to the depth word `tid` of buffer 2 holds.
kernel void fpc_recurse(device uint *out [[buffer(0)]], device const uint *depth [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
    op_t *f = (tid & 1u) != 0u ? fpc_rec_odd : fpc_rec_even;
    out[tid] = f(depth[tid]);
}

// The same choices among this module's functions, which are not visible: a select, a private
// array, and comparisons.
kernel void fpc_own_select(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    op_t *f = (tid & 1u) != 0u ? own_inc : own_dbl;
    out[tid] = f(tid);
}

kernel void fpc_own_private_array(device uint *out [[buffer(0)]],
                                  uint tid [[thread_position_in_grid]]) {
    op_t *fs[3] = {own_inc, own_dbl, own_sq};
    out[tid] = fs[tid % 3u](tid);
}

kernel void fpc_own_compare(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    op_t *fs[4] = {own_inc, own_dbl, own_sq, own_inc};
    op_t *f = fs[tid % 4u];
    op_t *g = fs[(tid / 4u) % 4u];
    uint r = 0u;
    r |= f == g ? 1u : 0u;
    r |= f == own_inc ? 2u : 0u;
    r |= f != nullptr ? 4u : 0u;
    r |= g == own_sq ? 8u : 0u;
    out[tid] = r;
}

// A 64-bit operation, so a declaration of a 64-bit type precedes the lifetime markers' and the
// decoded printout is the same under LLVM 21 and 22: the bit count of `tid` times 2^32 + 1.
kernel void fpc_wide(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = uint(popcount(ulong(tid) * 0x100000001ul));
}
