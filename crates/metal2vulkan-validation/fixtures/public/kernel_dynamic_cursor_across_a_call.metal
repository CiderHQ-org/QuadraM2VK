// A DYNAMIC byte cursor into a raw device buffer, crossing a helper call.
//
// The companion of `kernel_constant_cursor_across_a_call`, and the half that had no representation
// at all until `b321137c`. The emitter models a buffer RAW when the shader also views it as bytes,
// and then a non-zero GEP of it has no SPIR-V pointer value: it is a `Private` zero placeholder,
// with the real descriptor root and byte offset held only in the emitter's `raw_offsets`.
// `raw_device_call_arg_id` can carry a CONSTANT cursor onto the callee's parameter -- one offset
// per (callee, parameter) is all it can record -- but a runtime offset has nowhere to live, and
// Logical addressing forbids handing the callee the derived pointer instead. The only
// representation that keeps the callee's stores on the buffer is the one with no call boundary, so
// the translator splices this call into its caller in the AIR text and emits that.
//
// `noinline` is what keeps the helper in the AIR -- without it the frontend inlines it and there is
// no boundary to splice. The `uchar` view of the same buffer is what puts it on the raw model; a
// `float4`-only shader takes the typed access-chain path and never reaches this rule. `slot` is
// LOADED from the buffer, so the cursor is a runtime value the frontend cannot fold, and one thread
// is enough -- a `tid`-indexed cursor would fold to zero at `tid == 0` and prove nothing.
//
// The answer discriminates the cursor exactly. The helper writes slot `slot` == 2 and the entry
// writes slot 0, so a cursor lost to scratch leaves slot 2 holding its -1 sentinel; a cursor that
// collapsed to the buffer root would put the helper's vector in slot 0 before the entry overwrote
// it, which is the same visible answer, and either way slot 2 is the discriminator. Slots 1 and 3
// are written by nothing and hold -1, so an off-by-one cursor in either direction shows up too, and
// -1 is a value no arm can produce. `slot` and `seed` are read as BYTES -- byte 0 and byte 4 of the
// buffer -- so they are 2 and 5 exactly and no float rounding enters the answer; every written
// value is a small integer exactly representable in binary32. One thread, so nothing races.
#include <metal_stdlib>
using namespace metal;

__attribute__((noinline))
static void store_at(device float4* p, uint k)
{
    if (k > 0u) { *p = float4(float(k), float(k) + 1.0f, float(k) + 2.0f, float(k) + 3.0f); }
}

kernel void kernel_dynamic_cursor_across_a_call(
    device float4* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    device const uchar* bytes = (device const uchar*)out;
    uint slot = uint(bytes[0]);
    uint seed = uint(bytes[4]);
    store_at(out + slot, seed + tid);
    out[0] = float4(float(seed), float(slot), 0.0f, 0.0f);
}
