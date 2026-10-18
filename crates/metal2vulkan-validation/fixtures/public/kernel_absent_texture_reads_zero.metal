#include <metal_stdlib>
using namespace metal;

// On Metal (case 3fe49200), where this kernel reads the resource the pipeline does not provide,
// `dest` keeps its initial zero, and its write to that resource goes nowhere. The read's value is
// undefined, not zero: Metal's compiler drops a store of such a value, and its validation layer
// reads zero instead. So the engine refuses the observed read (`FunctionConstantArgument`), and the
// case's Metal rows stay as the record. `provide_absent` is false in the variant this fixture
// describes, so `absent_source` and `absent_sink` are exactly that resource -- and `present` and
// `dest` are the module's only other 2D read and only other 2D write, which is the shape under
// which a translator that stands "the module's only same-shaped texture" in for a lost handle reads
// and writes a texture the shader never named.
constant bool provide_absent [[function_constant(0)]];

kernel void absent_texture_reads_zero(
    texture2d<float, access::read> absent_source [[function_constant(provide_absent)]],
    texture2d<float, access::write> absent_sink [[function_constant(provide_absent)]],
    texture2d<float, access::read> present,
    texture2d<float, access::write> dest,
    uint2 gid [[thread_position_in_grid]]) {
  // The two reads are in separate lanes and name separate coordinates on purpose: given one
  // expression over one coordinate, Metal's own optimizer coalesces them onto a single handle and
  // BOTH answer zero, which measures the optimizer rather than the absent-resource rule.
  if (gid.x == 0 && gid.y == 0) {
    dest.write(present.read(uint2(0, 0)), gid);
  }
  if (gid.x == 1 && gid.y == 0) {
    dest.write(absent_source.read(uint2(0, 0)), gid);
  }
  if (gid.y == 1) {
    dest.write(float4(42.0f), gid);
  }
  // Issued last, so a store that failed to vanish is the value read back.
  absent_sink.write(float4(-1.0f), gid);
}
