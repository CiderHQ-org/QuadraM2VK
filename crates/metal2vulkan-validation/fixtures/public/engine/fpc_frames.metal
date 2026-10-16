// Owned MSL for Metal's frame per function:
// recursive visible functions whose frames Metal sizes differently, for its byte model. Each
// kernel `frames_*` calls its function at the depth word `tid` of buffer 2 holds (and the mode
// word 0 of buffer 3 holds, where it reads one) and writes the result to word `tid` of buffer 0;
// the pipeline links the functions the kernel names (`fr_*`). Every recursion ends at its input's
// depth. The shapes: many values live across the call (8 and 32 words of a hash chain), nested
// arrays ([4][4], [8][8] words, [3][2] of uint4), one-dimensional arrays of 8, 16, 32 and 64
// words written by a loop before the call and read after it, and pairs of unequal frames that
// alternate, the kernel entering at the small one, at the big one, or at either by a mode. The
// first kernel, `frames_wide` (a 64-bit bit count), declares a 64-bit type ahead of the lifetime
// markers, so the decoded printout is the same under LLVM 21 and 22.
#include <metal_stdlib>
using namespace metal;

kernel void frames_wide(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  out[tid] = uint(popcount(ulong(tid) * 0x100000001ul));
}

#define STEP(i) h = (h ^ (h >> 13)) * 0x85EBCA6Bu; uint a##i = h;
#define ADD(i) s += a##i * (i + 1u);

[[visible]] uint fr_live8(uint n) {
  if (n == 0u) { return 0u; }
  uint h = n * 0x9E3779B9u + 0x7F4A7C15u;
  STEP(0) STEP(1) STEP(2) STEP(3) STEP(4) STEP(5) STEP(6) STEP(7)
  uint s = fr_live8(n - 1u);
  ADD(0) ADD(1) ADD(2) ADD(3) ADD(4) ADD(5) ADD(6) ADD(7)
  return s;
}

[[visible]] uint fr_live32(uint n) {
  if (n == 0u) { return 0u; }
  uint h = n * 0x9E3779B9u + 0x7F4A7C15u;
  STEP(0) STEP(1) STEP(2) STEP(3) STEP(4) STEP(5) STEP(6) STEP(7)
  STEP(8) STEP(9) STEP(10) STEP(11) STEP(12) STEP(13) STEP(14) STEP(15)
  STEP(16) STEP(17) STEP(18) STEP(19) STEP(20) STEP(21) STEP(22) STEP(23)
  STEP(24) STEP(25) STEP(26) STEP(27) STEP(28) STEP(29) STEP(30) STEP(31)
  uint s = fr_live32(n - 1u);
  ADD(0) ADD(1) ADD(2) ADD(3) ADD(4) ADD(5) ADD(6) ADD(7)
  ADD(8) ADD(9) ADD(10) ADD(11) ADD(12) ADD(13) ADD(14) ADD(15)
  ADD(16) ADD(17) ADD(18) ADD(19) ADD(20) ADD(21) ADD(22) ADD(23)
  ADD(24) ADD(25) ADD(26) ADD(27) ADD(28) ADD(29) ADD(30) ADD(31)
  return s;
}

[[visible]] uint fr_nest44(uint n) {
  uint w[4][4];
  for (uint i = 0u; i < 4u; i++) { for (uint j = 0u; j < 4u; j++) { w[i][j] = n * 16u + i * 4u + j; } }
  uint s = n == 0u ? 0u : fr_nest44(n - 1u);
  for (uint j = 0u; j < 16u; j++) { s += w[(j * 7u + n) & 3u][(j * 5u + n) & 3u] * (j + 1u); }
  return s;
}

[[visible]] uint fr_nest88(uint n) {
  uint w[8][8];
  for (uint i = 0u; i < 8u; i++) { for (uint j = 0u; j < 8u; j++) { w[i][j] = n * 64u + i * 8u + j; } }
  uint s = n == 0u ? 0u : fr_nest88(n - 1u);
  for (uint j = 0u; j < 64u; j++) { s += w[(j * 7u + n) & 7u][(j * 5u + n) & 7u] * (j + 1u); }
  return s;
}

[[visible]] uint fr_nestv(uint n) {
  uint4 w[3][2];
  for (uint i = 0u; i < 3u; i++) { for (uint j = 0u; j < 2u; j++) { w[i][j] = uint4(n * 24u + i * 8u + j * 4u) + uint4(0u, 1u, 2u, 3u); } }
  uint s = n == 0u ? 0u : fr_nestv(n - 1u);
  for (uint j = 0u; j < 24u; j++) { s += w[(j + n) % 3u][(j * 5u + n) & 1u][(j * 3u + n) & 3u] * (j + 1u); }
  return s;
}

// A one-dimensional array of W words, written before the call and read after it.
#define WARR(W) \
  uint w[W]; \
  for (uint i = 0u; i < W; i++) { w[i] = n * W + i; }
#define WUSE(W) \
  for (uint j = 0u; j < W; j++) { s += w[(j * 7u + n) & (W - 1u)] * (j + 1u); }

[[visible]] uint fr_w8(uint n) { WARR(8u) uint s = n == 0u ? 0u : fr_w8(n - 1u); WUSE(8u) return s; }
[[visible]] uint fr_w16(uint n) { WARR(16u) uint s = n == 0u ? 0u : fr_w16(n - 1u); WUSE(16u) return s; }
[[visible]] uint fr_w32(uint n) { WARR(32u) uint s = n == 0u ? 0u : fr_w32(n - 1u); WUSE(32u) return s; }
[[visible]] uint fr_w64(uint n) { WARR(64u) uint s = n == 0u ? 0u : fr_w64(n - 1u); WUSE(64u) return s; }

// A pair of unequal frames: fr_pa keeps only n across its call, fr_pb a 16-word array. Mode 0
// alternates them, 1 recurses through fr_pb alone, 2 through fr_pa alone.
[[visible]] uint fr_pb(uint n, uint mode);
[[visible]] uint fr_pa(uint n, uint mode) {
  if (n == 0u) { return 0u; }
  uint s = mode == 2u ? fr_pa(n - 1u, mode) : fr_pb(n - 1u, mode);
  return n + s;
}
[[visible]] uint fr_pb(uint n, uint mode) {
  WARR(16u)
  uint s = n == 0u ? 0u : (mode == 0u ? fr_pa(n - 1u, mode) : fr_pb(n - 1u, mode));
  WUSE(16u)
  return s;
}

// A pair that both keep arrays: fr_qa 16 words, fr_qb 64; they alternate, the kernel entering
// at fr_qa.
[[visible]] uint fr_qb(uint n);
[[visible]] uint fr_qa(uint n) { WARR(16u) uint s = n == 0u ? 0u : fr_qb(n - 1u); WUSE(16u) return s; }
[[visible]] uint fr_qb(uint n) { WARR(64u) uint s = n == 0u ? 0u : fr_qa(n - 1u); WUSE(64u) return s; }

// Mutual pairs of an n-only function and a 64-word one: the kernel enters fr_sa only (the small
// one), fr_ta or fr_tb by its mode, and fr_ua only (the big one).
[[visible]] uint fr_sb(uint n);
[[visible]] uint fr_sa(uint n) { if (n == 0u) { return 0u; } return n + fr_sb(n - 1u); }
[[visible]] uint fr_sb(uint n) { WARR(64u) uint s = n == 0u ? 0u : fr_sa(n - 1u); WUSE(64u) return s; }
[[visible]] uint fr_tb(uint n);
[[visible]] uint fr_ta(uint n) { if (n == 0u) { return 0u; } return n + fr_tb(n - 1u); }
[[visible]] uint fr_tb(uint n) { WARR(64u) uint s = n == 0u ? 0u : fr_ta(n - 1u); WUSE(64u) return s; }
[[visible]] uint fr_ub(uint n);
[[visible]] uint fr_ua(uint n) { WARR(64u) uint s = n == 0u ? 0u : fr_ub(n - 1u); WUSE(64u) return s; }
[[visible]] uint fr_ub(uint n) { if (n == 0u) { return 0u; } return n + fr_ua(n - 1u); }

#define KERNEL(name, call) \
kernel void frames_##name(device uint *out [[buffer(0)]], device const uint *depth [[buffer(2)]], \
                          device const uint *mode [[buffer(3)]], uint tid [[thread_position_in_grid]]) { \
  out[tid] = call; \
}
KERNEL(live8, fr_live8(depth[tid]))
KERNEL(live32, fr_live32(depth[tid]))
KERNEL(nest44, fr_nest44(depth[tid]))
KERNEL(nest88, fr_nest88(depth[tid]))
KERNEL(nestv, fr_nestv(depth[tid]))
KERNEL(w8, fr_w8(depth[tid]))
KERNEL(w16, fr_w16(depth[tid]))
KERNEL(w32, fr_w32(depth[tid]))
KERNEL(w64, fr_w64(depth[tid]))
KERNEL(pair, (mode[0] == 1u ? fr_pb(depth[tid], mode[0]) : fr_pa(depth[tid], mode[0])))
KERNEL(quad, fr_qa(depth[tid]))
KERNEL(s, fr_sa(depth[tid]))
KERNEL(t, (mode[0] == 1u ? fr_tb(depth[tid]) : fr_ta(depth[tid])))
KERNEL(u, fr_ua(depth[tid]))
