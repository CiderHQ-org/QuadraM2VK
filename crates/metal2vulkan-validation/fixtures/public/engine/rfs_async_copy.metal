// air-rename: _Z8rfs_copymmPU14MTLthreadgroupvmmDv2_mPU9MTLdeviceKvmmDv2_lS4_i air.simdgroup_async_copy_2d.p3i8.p1i8
// air-rename: _Z14rfs_null_eventv air.get_null_simdgroup_event
// air-rename: _Z11rfs_is_nullP9rfs_event air.is_null_simdgroup_event
// air-rename: _Z8rfs_waitiPP9rfs_event air.wait_simdgroup_events
// Owned fixture: a simdgroup's asynchronous 2D copy from device to threadgroup memory
// and its events, which MPS's matrix kernels call and MSL spells only in a private header. The
// kernels call `rfs_copy`, `rfs_null_event`, `rfs_is_null` and `rfs_wait`, which this source
// declares and never defines, and the directives above rename every call to the intrinsics in both
// companions (scripts/regen-air-fixtures).
#include <metal_stdlib>
using namespace metal;

struct rfs_event;
thread rfs_event *rfs_copy(ulong size, ulong align, threadgroup void *dst, ulong dst_epr,
                           ulong dst_stride, ulong2 dst_tile, const device void *src, ulong src_epr,
                           ulong src_stride, long2 src_tile, long2 offset, int clamp);
thread rfs_event *rfs_null_event();
bool rfs_is_null(thread rfs_event *e);
void rfs_wait(int n, thread rfs_event **events);

#define TG_BYTES 4096

// One threadgroup fills TG_BYTES of threadgroup memory with 0xee, then every simdgroup copies the
// tile `p` describes into it and waits:
// p: [0] dst_epr [1] dst_stride [2, 3] dst_tile [4] src_epr [5] src_stride [6, 7] src_tile
// [8, 9] offset [10] dst byte offset [11] src byte offset [12] the byte offset between the
// simdgroups' destinations.
// out: the TG_BYTES / 4 words of threadgroup memory after a threadgroup barrier, then per
// simdgroup s word TG_BYTES / 4 + 2s the copy's event's `is_null` and + 2s + 1 the null event's.
template <typename T>
static void copy_kernel(threadgroup uchar *tg, device const uchar *src, device uint *out,
                        constant long *p, uint lid, uint sg, uint lane, uint ntg) {
  for (uint i = lid; i < TG_BYTES; i += ntg) {
    tg[i] = 0xee;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  thread rfs_event *none = rfs_null_event();
  bool none_null = rfs_is_null(none);
  long dst_off = p[10] + long(sg) * p[12];
  thread rfs_event *ev = rfs_copy(sizeof(T), alignof(T), (threadgroup void *)(tg + dst_off),
                                  ulong(p[0]), ulong(p[1]), ulong2(ulong(p[2]), ulong(p[3])),
                                  (const device void *)(src + p[11]), ulong(p[4]), ulong(p[5]),
                                  long2(p[6], p[7]), long2(p[8], p[9]), 0);
  bool ev_null = rfs_is_null(ev);
  rfs_wait(1, &ev);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup uint *t = (threadgroup uint *)tg;
  for (uint i = lid; i < TG_BYTES / 4; i += ntg) {
    out[i] = t[i];
  }
  if (lane == 0) {
    out[TG_BYTES / 4 + 2 * sg] = ev_null ? 1u : 0u;
    out[TG_BYTES / 4 + 2 * sg + 1] = none_null ? 1u : 0u;
  }
}

kernel void rfs_async_copy_1(device const uchar *src [[buffer(0)]], device uint *out [[buffer(1)]],
                             constant long *p [[buffer(2)]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup uchar tg[TG_BYTES];
  copy_kernel<uchar>(tg, src, out, p, lid, sg, lane, ntg);
}

kernel void rfs_async_copy_2(device const uchar *src [[buffer(0)]], device uint *out [[buffer(1)]],
                             constant long *p [[buffer(2)]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup uchar tg[TG_BYTES];
  copy_kernel<ushort>(tg, src, out, p, lid, sg, lane, ntg);
}

kernel void rfs_async_copy_4(device const uchar *src [[buffer(0)]], device uint *out [[buffer(1)]],
                             constant long *p [[buffer(2)]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup uchar tg[TG_BYTES];
  copy_kernel<uint>(tg, src, out, p, lid, sg, lane, ntg);
}
