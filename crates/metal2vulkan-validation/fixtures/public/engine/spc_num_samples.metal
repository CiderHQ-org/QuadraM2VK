// The pipeline's rasterization sample count. `get_num_samples()` is the count the
// render pipeline was created with: Metal's header spells it `__metal_get_num_samples(0)`, AIR's
// `air.get_num_samples.i32` with operand 0. The engine reads the `SampleCount` specialization
// constant, which the binder sets to the pipeline's `rasterizationSamples`.
//
// A case draws the generated fullscreen triangle through this function into an
// `r32_uint` target: every covered pixel holds the count, 1 for the single-sample pipeline the
// case schema authors.
#include <metal_stdlib>
using namespace metal;

fragment uint spc_num_samples() {
    return get_num_samples();
}
