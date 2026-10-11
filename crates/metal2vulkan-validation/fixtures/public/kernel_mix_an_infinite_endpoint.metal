// Metal's `mix(x, y, t)` is the blend `x * (1 - t) + y * t`, evaluated -- not a branch on `t`. At a
// finite pair that distinction is invisible, because the blend already returns exactly `x` at
// `t == 0` and exactly `y` at `t == 1`. At a NON-finite pair it is the whole answer: `inf * 0` is
// NaN, so an infinite or NaN endpoint poisons the result even at the `t` where it is supposedly
// weighted out.
//
// This kernel exists because the lowering used to disagree. It wrapped `FMix` in
// `select(t == 0, x, ...)` and `select(t == 1, y, ...)`, which answered 5 for `mix(inf, 5, 1)`
// where Metal answers NaN, and `-0` for `mix(1, -0, 1)` where Metal answers `+0`. Eight of ten
// probed endpoint rows were wrong, and the two it got right were the two the bare blend already
// had.
//
// Every operand is read from the buffer: a literal infinity is folded by the frontend and the
// kernel stops testing the hardware. The NaN rows are reported through `isnan` rather than by
// their bits, since a NaN payload is not something any implementation owes.
//
// Device-measured on Apple M3 Max / macOS 26.5.2:
//   mix(inf, 5, 1) = mix(5, inf, 0) = mix(NaN, 5, 1) = mix(5, NaN, 0) = NaN
//   mix(inf, inf', 1) = NaN for two distinct loads, but mix(v, v, 1) = v for one
//   mix(1, -0, 1) = +0   (the blend's own `-0` is not what Metal answers)
//   mix(3, 7, 1) = 7, mix(3, 7, 0) = 3, mix(2, 4, 0.5) = 3   -- the ordinary blend still works
#include <metal_stdlib>
using namespace metal;

kernel void kernel_mix_an_infinite_endpoint(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]])
{
    float infinity = in[0];
    float not_a_number = in[1];
    float second_infinity = in[12];
    float five = in[2];
    float zero_t = in[3];
    float one_t = in[4];
    float negative_zero = in[5];
    float one = in[6];
    float three = in[7];
    float seven = in[8];
    float half_t = in[9];
    float two = in[10];
    float four = in[11];

    // An endpoint that is weighted out still poisons the blend.
    out[0] = isnan(mix(infinity, five, one_t)) ? 1.0f : 0.0f;
    out[1] = isnan(mix(five, infinity, zero_t)) ? 1.0f : 0.0f;
    out[2] = isnan(mix(not_a_number, five, one_t)) ? 1.0f : 0.0f;
    out[3] = isnan(mix(five, not_a_number, zero_t)) ? 1.0f : 0.0f;
    // Two DISTINCT loads, both infinite. Passing one value twice does not test this: measured,
    // `mix(v, v, 1)` on a single loaded `v` answers `v`, not NaN, because with the endpoints
    // symbolically equal the blend collapses before either infinity meets a zero weight.
    out[4] = isnan(mix(infinity, second_infinity, one_t)) ? 1.0f : 0.0f;

    // The signed-zero endpoint, where the guard also answered differently.
    out[5] = mix(one, negative_zero, one_t);

    // The ordinary blend, so the rows above prove something.
    out[6] = mix(three, seven, one_t);
    out[7] = mix(three, seven, zero_t);
    out[8] = mix(two, four, half_t);
}
