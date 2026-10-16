// Owned fixture: fragment inputs four function constants gate, each read
// without a guard into one lane of the color output: a varying (MSL 5.8.1.3), a color input
// (framebuffer fetch), `[[primitive_id]]`, and `[[barycentric_coord]]` (MSL 5.8.1.2). With a
// constant false (or undefined) its input is not declared and the language gives its read no
// value; the engine reads zero and declares neither the input nor what reading it would need
// (a fetch, the `PrimitiveId` or `BaryCoordKHR` builtin). With a constant true its input is an
// ordinary one. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool has_tint [[function_constant(0)]];
constant bool has_prior [[function_constant(1)]];
constant bool has_primitive [[function_constant(2)]];
constant bool has_barycentric [[function_constant(3)]];

struct FcaVaryings {
    float4 position [[position]];
    float4 tint [[function_constant(has_tint)]];
};

fragment float4 fca_fragment_absent_inputs(
    FcaVaryings in [[stage_in]],
    float4 prior [[color(0), function_constant(has_prior)]],
    uint primitive [[primitive_id, function_constant(has_primitive)]],
    float3 barycentric [[barycentric_coord, function_constant(has_barycentric)]]) {
  return float4(in.tint.x, prior.y, barycentric.z, float(primitive));
}
