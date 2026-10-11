// A constant byte cursor into a raw device buffer, crossing a helper call.
//
// The emitter models a buffer RAW when the shader also views it as bytes, and then a non-zero GEP
// of it has no SPIR-V pointer value at all: it is a `Private` zero placeholder, and the real
// descriptor root plus byte offset live in the emitter's `raw_offsets`. Handing that to a helper
// would send the helper's stores to scratch, so `raw_device_call_arg_id` instead passes the ROOT
// and records the cursor against the callee's parameter, which the callee then re-applies. That
// re-application is the thing no device case had measured: it is arithmetic on a byte offset
// performed in one function and consumed in another, and getting it wrong writes the right value
// into the wrong slot rather than failing.
//
// `noinline` is what keeps the helper in the AIR -- without it the frontend inlines it and the
// cursor never crosses anything. The `uchar` view of the same buffer is what puts it on the raw
// model; a `float4`-only shader takes the typed access-chain path and never reaches this rule.
//
// The answer discriminates the cursor exactly. The helper writes `out + 2` and the entry writes
// `out[0]`, so a dropped cursor writes slot 0 twice: slot 2 would keep its -1 sentinel and slot 0
// would hold the helper's vector before the entry overwrote it. Slots 1 and 3 are never written by
// anything and hold -1, so a cursor off by one slot in either direction is visible too. Every
// value is a small integer exactly representable in binary32, and `seed` is read as a BYTE (byte 4
// of the buffer, which is the low byte of `out[0].y`), so no float rounding enters the answer.
#include <metal_stdlib>
using namespace metal;

__attribute__((noinline))
static void store_at(device float4* p, uint k)
{
    if (k > 0u) { *p = float4(float(k), float(k) + 1.0f, float(k) + 2.0f, float(k) + 3.0f); }
}

kernel void kernel_constant_cursor_across_a_call(
    device float4* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    device const uchar* bytes = (device const uchar*)out;
    uint seed = uint(bytes[4]);
    store_at(out + 2, seed + tid);
    out[0] = float4(float(seed), 0.0f, 0.0f, 0.0f);
}
