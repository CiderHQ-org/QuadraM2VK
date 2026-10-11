// A vertex entry that reads [[amplification_id]] and writes [[render_target_array_index]] in the
// same shader. 27 corpus vertex modules do both; this is the smallest shader that does, so it is
// the one that says whether MoltenVK can run the pair.
#include <metal_stdlib>
using namespace metal;

struct VertexAmplificationLayerOut {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
    float4 color [[user(locn0)]];
};

vertex VertexAmplificationLayerOut vertex_amplification_id_layer(
    uint vid [[vertex_id]],
    ushort amp [[amplification_id]])
{
    float2 corner = float2(float((vid << 1) & 2u), float(vid & 2u)) * 2.0f - 1.0f;
    VertexAmplificationLayerOut out;
    out.position = float4(corner, 0.0f, 1.0f);
    out.layer = uint(amp);
    out.color = float4(float(amp) + 1.25f, 2.5f, 3.75f, 5.0f);
    return out;
}
