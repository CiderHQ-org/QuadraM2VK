// A device pointer that reaches its use inside a struct field, read back by an OUT-OF-LINE helper.
// The helper is emitted as its own SPIR-V function, so at emit time the pointer is only a Private
// byte placeholder -- the store that gave it an address lives in the caller -- and every byte-view
// load in the helper has to be deferred and repaired after inlining.
//
// The record is float3 at +0 and a lone float at +12, both read through the same `device const
// uchar*` byte view at the same dynamic base. That pairing is the point: the vector load and the
// scalar load are the same byte assembly at the same address, and the emitter used to serve only
// the vector one -- `emit_scalar_slots_to_wider_load` accepted a vector result and declined a
// scalar, so the scalar returned "cannot reinterpret load of byte pointer to Float". Ten corpus
// sources failed on exactly that. Writing both into one float4 makes the two paths compare against
// each other on the device: if the scalar assembly reads different bytes than the vector one, or
// reads Private scratch instead of the buffer, lane 3 disagrees with lanes 0-2's neighbourhood.
#include <metal_stdlib>
using namespace metal;

struct record_view {
    device const uchar* bytes;
};

__attribute__((noinline))
static float4 read_record(thread const record_view& v, uint byte_offset)
{
    device const uchar* base = v.bytes + byte_offset;
    float3 head = *reinterpret_cast<device const packed_float3*>(base);
    float tail = *reinterpret_cast<device const float*>(base + 12);
    return float4(head, tail);
}

kernel void kernel_byte_view_scalar_through_helper(
    device const uchar* src [[buffer(0)]],
    device float4* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    record_view v;
    v.bytes = src;
    out[gid] = read_record(v, gid * 16);
}
