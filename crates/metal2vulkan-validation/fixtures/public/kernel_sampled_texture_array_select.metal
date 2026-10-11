// A dynamically indexed `metal::array` of SAMPLED textures, read through one sampler.
//
// Logical SPIR-V has no pointer-to-image, so the emitter cannot name the handle the index picks.
// It has to collect the whole store set into the alloca, replay it as a select tree over the three
// image ids, and prove every arm is available at the sample site. `native/opaque_image_select.rs`
// has two halves for that -- one for a directly read image, one for a SAMPLED image, which is a
// different SPIR-V value (`OpSampledImage`, not `OpImage`) and needs its own image-type recovery.
// The direct half was authored on a corpus source; this is the sampled half, and the corpus has
// exactly one host for it (an 870-line tone mapper). Measured gain over every cased source: 18
// functions, being `plan_tree`, `replay_selected_image`, `sampled_image_image_type`,
// `is_sampled_image_value_op`, `tree_values_available_at` and `value_available_at` with their
// closures, plus `collect_select_tree` and `fresh`.
//
// All three arms are live in one dispatch of three threads, and `which` is (2, 0, 1) rather than
// the identity, so a select tree wired in the wrong order answers differently instead of silently
// agreeing. `% 3` keeps the index in range without letting the frontend fold it -- the value comes
// from a buffer.
//
// Every answer is an exact integer. Texture k's texel (x, y) carries the four channels
// 100(k+1) + 10(2y + x) + c for c in 1..4, and the shader folds them as r + 10g + 100b + 1000a.
// The sampler is `nearest`, `clamp_to_edge`, normalized, so (0.25, 0.75) on a 2x2 image resolves
// to texel (0, 1) exactly with no filtering: 0.25 * 2 = 0.5 and 0.75 * 2 = 1.5 both floor into
// their own texel and neither sits on a boundary. Every term and every partial sum is an integer
// below 2^24, so `fast` reassociation cannot move the result. The three answers are 359841, 137641
// and 248741: the texture index sets the hundred-thousands digit and the texel index sets the tens,
// so a wrong image and a wrong texel are separately visible.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_sampled_texture_array_select(
    texture2d<float> t0 [[texture(0)]],
    texture2d<float> t1 [[texture(1)]],
    texture2d<float> t2 [[texture(2)]],
    sampler samp [[sampler(0)]],
    device const uint* which [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    array<texture2d<float>, 3> tex = { t0, t1, t2 };
    uint i = which[tid] % 3u;
    float4 v = tex[i].sample(samp, float2(0.25f, 0.75f));
    out[tid] = v.x + 10.0f * v.y + 100.0f * v.z + 1000.0f * v.w;
}
