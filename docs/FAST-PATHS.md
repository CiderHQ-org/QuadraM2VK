# Fast paths: the accuracy ledger

**An accuracy note is logged, never fatal.** The work keeps running; only what Metal itself refuses
ends it.

Each row of the first table is one place where the default trades exactness against Metal for
speed: pipeline-creation ("making") time or GPU time, mostly off Apple's GPUs. The fast form is the
default, the accurate form stays behind its switch, and both are kept. This file is a ledger, not a
journal: rows are edited in place, with no history. A change that adds, removes or changes a fast
default, its bound or its switch updates its row in the same commit. The test `tests::fast_paths`
fails when a `DevicePolicy` accuracy choice, one of its values, or a precision note a fast default
records is missing here, or when this file names a policy value that no longer exists.

Every switch is a field of `DevicePolicy` (the runtime's `DeviceOptions::policy`). On MoltenVK on
an Apple GPU, whose sampler and `float` functions are Metal's own, most rows do not apply. Each
difference is recorded in the pipeline's artifact as a `PrecisionNote` and announced as
`metal2vulkan-degrade-<code>` when the runtime makes the pipeline state; each note's rustdoc and
docs/API.md section 9 state its bound.

## Fast defaults

Costs are measured on the RTX 5080 Laptop GPU (NVIDIA 615.71.09) with `scripts/making-bench`:
cold, the command queue made first, the driver's disk cache off; the making of the state's
pipeline, then GPU time per draw, the default against the accurate form alone. The workloads are
three consumer pipelines: A2X (`a2x-s4`, `a2x-s96`), Xgc14 (`xgc14-bgra8`) and Xgc15
(`xgc15-rgba8`). The switches were measured at master e4b1bdf5e, half
rounding at master 1ee783c77, whose default output is the same;
a cell that names another source took its numbers there.

| What it is | Default (fast) | Accurate switch | Recorded bound (note, code) | Cost of the accurate form (5080) |
|---|---|---|---|---|
| Samples, gathers and depth comparisons go to the device's own sampler: its filter, footprint, level, anisotropy, sRGB decode and comparison. | `sampling = SamplingPolicy::NativeDevice` | `sampling = SamplingPolicy::Exact`: Metal's sampler emulated by fetches, a pipeline per bound sampler state | Linear filter within one code (1/255) on the measured devices, sRGB within 1.83; levels, anisotropy, cubes, 3D, gathers and comparisons unmeasured. `DeviceSampling` (`device-sampling`) | Not measured on today's master. At the switch (maths and texel conversions still exact): Xgc15 making 24.2-24.8 s exact against 12.5-13.9 s native |
| A texel's `float` value is the device's conversion of its codes; a `half` result is the device's `float` texel rounded to nearest even. | `texel_conversion = TexelConversionPolicy::Device` | `texel_conversion = TexelConversionPolicy::Exact`: Metal's conversion by the bound format's class of every read; a sample's or gather's also with `sampling = SamplingPolicy::Exact` (under native sampling the device's sampler decodes it) | One `f32` ulp for 8- and 16-bit codes, 0.18 of a code for 5- and 6-bit packed channels; `half` within one ulp; NaN bits the device's. `DeviceTexelConversion` (`device-texel-conversion`), `HalfTexelConversion` (`half-texel-conversion`, also recorded for heap textures and unmeasured formats) | Exact reads (the same reads): Xgc14 0.88-0.89 → 0.90-0.93 s, GPU 0.039 → 0.046 ms; Xgc15 0.65-0.66 → 0.68 s, 0.090 → 0.094 ms; A2X unchanged (no affected read). Samples: with the sampling row |
| The maths library (`sin`, `exp`, `log`, `pow`, `atan2`...), divisions and Metal's fast reciprocal, and a fused multiply-add without `OpFmaKHR` are the device's instructions, not routines of Metal's answers. | `float_maths = FloatMathsPolicy::Native` | `float_maths = FloatMathsPolicy::Exact` | Vulkan's precision table (`Sin` 2^-11 absolute in [-π, π], `Exp` 3 + 2·\|x\| ulp, `Atan` 4096 ulp) against Metal's 4-6 ulp; division 2.5 ulp; `fma` one `half` ulp (a `float` one may round twice). `NativeTranscendentals` (`native-transcendentals`), `NativeDivision` (`native-division`), `NativeFma` (`native-fma`) | A2X 1.00-1.02 → 4.61-4.67 s, GPU 0.098 → 0.194 ms; Xgc14 0.88-0.89 → 3.55-3.58 s, 0.039 → 0.095-0.099 ms; Xgc15 0.65-0.66 → 1.94-2.00 s, 0.090 → 0.176 ms |
| A NaN result's bits are the device's (no select of Metal's NaN), and `fmin`, `fmax`, `min3`, `max3`, `clamp` and `saturate` are the device's `NMin`, `NMax` and `NClamp`. | `nan_and_min_max = NanMinMaxPolicy::Native` | `nan_and_min_max = NanMinMaxPolicy::Exact` | A NaN where Metal answers one, its bits the device's; the operand Metal picks but at two NaNs, `+0` against `-0`, and `f32` denormals. `NativeNanEncoding` (`native-nan-encoding`), `NativeMinMax` (`native-min-max`) | A2X 1.00-1.02 → 1.85-1.88 s, GPU 0.098 → 0.116 ms; Xgc14 0.88-0.89 → 1.59-1.63 s, 0.039 → 0.048 ms; Xgc15 0.65-0.66 → 1.25-1.28 s, 0.090 → 0.102 ms |
| `dot`, `mix` and `fast::sqrt` are the device's `OpDot`, `FMix` and `Sqrt`, not Metal's fused chain, its two fused multiply-adds and `x · rsqrt(x)`. | `rounding_order = RoundingOrderPolicy::Native` | `rounding_order = RoundingOrderPolicy::Exact` | `float` `dot` of n lanes within (3n − 1)·2^-24·Σ\|aᵢbᵢ\|, `mix` within 2^-21 of its two terms, `fast::sqrt` 4.5 ulp. `NativeDot` (`native-dot`), `NativeMix` (`native-mix`), `NativeTranscendentals` (`native-transcendentals`) | A2X 1.00-1.02 → 1.03-1.06 s, GPU 0.098 → 0.101 ms; Xgc14 0.88-0.89 → 0.97-1.02 s, 0.039 → 0.041 ms; Xgc15 0.65-0.66 → 0.73 s, 0.090 → 0.088 ms |
| A `half` result computed in `float` rounds through the device's conversion to 16 bits (three instructions), or its `OpQuantizeToF16` without `shaderFloat16`, not Metal's rounding in integer arithmetic (about twenty). | `half_rounding = HalfRoundingPolicy::Native` | `half_rounding = HalfRoundingPolicy::Exact` | One `half` ulp; 65504 where Metal answers infinity on a device rounding toward zero; a half denormal may flush; NaN bits and a zero's sign the device's. Or no rounding at all, measured: NVIDIA (RTX 3090, 615.71.09) drops the conversion pair, and lavapipe does in arithmetic on Mesa 26.2.4 with LLVM 23.1.1 and for a loaded texel on Mesa 26.0.8 with LLVM 21.1.8, whose arithmetic measured rounded, so the result keeps `float` precision, its differences compounding across operations, and a value past `half`'s range stays finite until a `half` store saturates it to infinity; RADV and ANV take their quantize, measured to be Metal's (exact, no note); the RTX 5080 Laptop GPU (615.71.09) rounds, measured: 8192 of 8192 narrowed sums are Metal's. `NativeHalfConversion` (`native-half-conversion`), `NativeHalfQuantize` (`native-half-quantize`) | A2X 0.99-1.04 → 7.19-7.21 s, GPU 0.098-0.100 → 0.139-0.140 ms (+40%); Xgc14 0.86-0.87 → 0.87-0.92 s, Xgc15 0.63-0.66 → 0.69-0.70 s, GPU unchanged (RTX 5080). Rounding natively by `OpQuantizeToF16` instead, which the RTX 3090 and lavapipe honour (Metal's answer on every operand tested), was measured and declined: A2X 1.3 s → 18-22 s, GPU 0.14 → 1.5 ms on an RTX 3090 |
| A `half` read from its bits (a load, a reinterpretation, a texel's bits) outside every loop is widened by the device (`UnpackHalf2x16`), not decoded exactly. | `half_rounding = HalfRoundingPolicy::Native` | `half_rounding = HalfRoundingPolicy::Exact` (with the row above) | A half denormal may read as zero where the device flushes them; NaN payload the device's. `NativeHalfConversion` (`native-half-conversion`) | Not measured alone on today's master. At its merge: A2X 1.04-1.07 s with it against 1.27-1.28 s with the exact decode; Xgc unchanged |
| The `half` conversion converts the value itself (no `NoContraction` barrier), so the device's compiler may fold it into the instruction computing the value, rounding once. | `half_rounding = HalfRoundingPolicy::Native` | `half_rounding = HalfRoundingPolicy::Exact` (with the rows above) | The row above's: on NVIDIA (RTX 3090), and on lavapipe for a loaded texel (Mesa 26.0.8), the pair is dropped whether or not the barrier stands, so the barrier does not restore the rounding there. `NativeHalfConversion` (`native-half-conversion`) | At its merge: A2X 0.99-1.02 s without the barrier against 1.00-1.05 s with it; Xgc within a round's spread |
| A `half` texel's NaN is the rounding's own NaN, not the texel's bits (the keep's four instructions a texel are left out). | `half_rounding = HalfRoundingPolicy::Native` and `nan_and_min_max = NanMinMaxPolicy::Native` | `nan_and_min_max = NanMinMaxPolicy::Exact` (the keep under native half rounding, master ecd39568bb); `half_rounding = HalfRoundingPolicy::Exact` also keeps it, at that row's cost. The same switch also gives Metal's full rule: `0x7e00` for every NaN but a one-texel read, nearest sample or gather of a 16-bit float texture, by the slot's format class read as data, a heap handle's class index and a runtime sampler's filter. A gather with a `half` result fetches its four texels where the class is a half-float one, since NVIDIA's gather hands back every NaN as `0x7fffffff`. **Not yet exact:** a cube gather, bicubic and emulated linear, fetched or compared samples. Possible, not built: the runtime sampler's filter read as push data rather than specialized, which would spare the remake below. Measured on the RTX 5080: under the switch no `half` lane of the runtime lane's texel reads, samples and gathers differs from Metal's (1,469 differ by default). Under the switch `DeviceNanEncoding` is still announced; the consumer logs notes, so no device fact is built to drop it | NaN bits the device's (`0x7FFF` on the RTX 5080), where Metal hands back a 16-bit float texel's own bits on a read, nearest sample or gather, and `0x7e00` for a 32-bit or small float texel's NaN and a linear sample's. `NativeHalfConversion` (`native-half-conversion`); the read also records `DeviceNanEncoding` (`device-nan-encoding`) | Measured on top of the NaN switch's own row (its probe with the keep, c0a814a13, against its probe without, 3e8a01c54): A2X 341,299 → 347,519 words, 1.82-1.86 → 1.81-1.87 s, GPU 0.115-0.117 → 0.115-0.116 ms, within noise; Xgc14 and Xgc15 read no `half` texel (words unchanged). The full rule (its probe a5c642ebe against the same tip without it, 851650c81): A2X 347,519 → 372,429 words (+7.2%), creation 1.85-1.88 → 1.99-2.04 s; where a linear runtime sampler is bound at the draw the sampler word's filter is now specialized, so the first draw remakes (1.94-1.96 s) and its GPU time rises 0.115-0.117 → 0.124-0.125 ms; with a nearest sampler GPU 0.115-0.116 → 0.115-0.117 ms; Xgc14 and Xgc15 words unchanged. On the RTX 3090 the same pair: A2X creation 2.91-3.06 → 3.17-3.23 s, the first draw's remake 3.0 s and its GPU time 0.17-0.18 → 0.79-1.15 ms, with a nearest sampler 0.17-0.18 ms both |
| A fragment function without fast math returns a `half` color to a blended attachment: the device blends in its own arithmetic and rounds once, where Metal blends a 16-bit float attachment in `half` arithmetic (each factor, product and sum rounded to `half`). | `half_blend = HalfBlendPolicy::Device` | `half_blend = HalfBlendPolicy::Exact`: the function computes Metal's blend in the interlocked storage form, for a `half4` output into `R16Float`, `RG16Float` or `RGBA16Float` where that form is exact for the pipeline (RADV, ANV, NVIDIA, MoltenVK); elsewhere, and for lavapipe, 32-bit float, 8-bit, a module naming no math mode or a blend of the second source, the device's blend and its note stay | 2.5 `half` ulps at the largest operand on NVIDIA (the blend color and one minus it), two on RADV, ANV and lavapipe (16-bit float); two `float` ulps (32-bit float, one on NVIDIA); one or two codes (8-bit, extended range); MoltenVK stores Metal's bytes. Measured with and without the factors' magnitudes at 0bcc0163e5. `HalfOutputPrecision` (`half-output-precision`) | A probe at 924fc11603 (a `half4` over a 1920x1080 `RGBA16Float` target, One / OneMinusSourceAlpha): making 11.1 → 24.0 ms; GPU 25 → 82 us one draw, 506 → 2291 us 32 overlapping draws (the interlock orders them). RTX 3090 (the same draw at 50a065eb98): making 12.9 → 27.4 ms; GPU 45 → 115 us one draw, 648 → 3268 us 32 overlapping draws |
| A computed `half` or `float` color blended into an extended-range attachment (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`) rounds the blend's source as a loaded one; Metal's compiler rounds it by how it computed it. Every device. | The in-shader blend's loaded rounding (no switch) | **Not built yet:** Metal's rounding per shape is probed, not modelled | One code. `ExtendedRangeBlendSource` (`extended-range-blend-source`), `ExtendedRangeFloatBlendSource` (`extended-range-float-blend-source`) | Not measured |
| A bicubic sample of an 8- or 16-bit normalized texture sums its taps exactly and rounds half up; Metal truncates its row terms, row sums and column terms first. Every device. | The exact sum (no switch) | **Not built yet:** Metal's datapath is pinned (17.75M lanes measured, no miss); its emitter is not written | One grid step (1/4080 at 8 bits) within about 2^-21 of a tie. `BicubicFilterArithmetic` (`bicubic-filter-arithmetic`) | Not measured (Metal's form is expected no larger) |

All at once: `DevicePolicy::exact_answers()` sets every switch above but sampling's (validation
adds `sampling = SamplingPolicy::Exact`). Against master 1ee783c77 it cost A2X 0.99-1.04 → 17.5 s, GPU 0.098 → 0.267 ms; Xgc14 0.86-0.87 →
4.95-5.01 s, 0.039 → 0.105-0.110 ms; Xgc15 0.63-0.66 → 3.06-3.15 s, 0.090 → 0.194-0.197 ms.

Speed choices that change no answer (the roll pass's floor; a specialization measured free) are not
rows.

## Not speed choices: device gaps

No switch gives Metal's answer here: the device lacks what an exact form needs, or no exact form is
known. The rest of the differences are docs/API.md section 9's precision notes.

| What it is | Why there is no switch | Note, code |
|---|---|---|
| `f32` denormals in operations left to the device (a remainder, a `double` conversion, `dot`'s fused sum, a function no routine computes): Metal reads them as zero. | Vulkan lets each device keep or flush them. `+ - *`, comparisons, `fma`, `sqrt`, `rsqrt`, `floor`, `ceil` and divisions read them as Metal on every device. | `DeviceDenormals` (`device-denormals`) |
| A denormal 32-bit float texel the device's sample or gather reads as zero (RADV's gathers and nearest samples, NVIDIA's gathers, ANV's nearest samples), where the module cannot fetch the texels itself. | The device's filter decides; no fetch form reaches filtered, cube, heap or runtime-sampler samples. | `DeviceTexelDenormals` (`device-texel-denormals`) |
| Depth values outside [0, 1] stored clamped (RADV, ANV) on a device without `VK_EXT_depth_range_unrestricted`. | Vulkan leaves them undefined without the extension: enable it. | `RestrictedDepthRange` (`restricted-depth-range`) |
| Signed zeros, infinities and NaNs where the device lacks `SignedZeroInfNanPreserve` at the operation's width. | A device capability. | `DeviceSignedZeroInfNan` (`device-signed-zero-inf-nan`) |
| NVIDIA's interpolation rounds an exactly representable varying one `f32` ulp toward zero. | The exact path needs Metal's interpolation formula, which is not measured. | `DeviceInterpolation` (`device-interpolation`) |
| A fast-math function's `half` color into a blend: within three `half` ulps (16-bit float) or two `float` ulps (32-bit float). | Metal's compiler regroups the blend per pipeline; no form the function and the blend state decide is Metal's. | `HalfOutputFastMathBlend` (`half-output-fast-math-blend`) |
| Float simdgroup reductions sum in the device's association order. | Apple's order is not modelled. | `ReductionOrder` (`reduction-order`) |

## Accuracy switches whose default is Metal's

- `unorm8_linear_filter = Unorm8LinearFilterPolicy::Exact` (the default) keeps Metal's filter for
  an 8-bit unsigned normalized texture's linear sample where Metal's sampler is emulated
  (`sampling = SamplingPolicy::Exact`). `unorm8_linear_filter =
  Unorm8LinearFilterPolicy::DeviceWithinOneCode` hands it to a device measured within one code
  (`DeviceUnorm8LinearFilter`, `device-unorm8-linear-filter`): it bought no GPU time on the
  consumer's fragments and cost module size and making time.
- `framebuffer_fetch = FramebufferFetchPolicy::RequireOrdered` (the default) refuses a framebuffer
  fetch without rasterization-order attachment access; `framebuffer_fetch =
  FramebufferFetchPolicy::AcceptUnordered` accepts it unordered (`UnorderedFramebufferFetch`,
  `unordered-framebuffer-fetch`). A coverage choice, not a speed one.
- `pointer_domain`, `handle_domain`, `containment` and `isolation` choose how resources resolve and
  how guests are contained, not an answer of Metal's arithmetic or sampling.

## Known gaps

Open differences from Metal that no note announces yet, and runtime costs still above the
driver's. Each is open work.

- **Tininess before rounding.** Metal flushes a `float` result whose exact value lies below
  `2^-126` even where rounding would reach `2^-126`: `fma(2^-126, -2^-126, 2^-126)` is `0` on Metal
  and on MoltenVK, `2^-126` on RADV, ANV, NVIDIA and lavapipe. An exact form needs a scaled check on
  every `float` multiply and `fma`; it is not built.
- **A fragment sample's bias within the device's limit.** Off MoltenVK the shader forms Metal's
  bias, `clamp(floor(half(bias) · 64) / 64, -32, 31.984375)`, but the device's own level arithmetic
  applies it, unannounced.
- **Simdgroup barriers in divergent code** stay where they are, as SPIR-V defines them (they wait
  for their tangle). No case compares them with Metal.
- **Commit to completion.** A command buffer takes longer from commit to completion through the
  runtime than a bare driver submission: on a quiet Radeon 780M (RADV) an empty command buffer took
  51.6 µs against a bare barrier's 45.1 µs. On MoltenVK most of the rest is the two GPU timestamps
  each timed command buffer writes: on an M4 a bare barrier took 14.7 µs alone and 167.8 µs with
  them. `DeviceOptions::gpu_times_from_host` (off by default, announced) answers GPU times from the
  host clock and spares them.
- **A view of another format without `PixelFormatView` usage** is a shadow image whose texels are
  copied at each use (`shadowed-texture-view`): about 50 to 95 µs a command buffer for a 64x64
  texture, measured on the M3 Max and M4 (MoltenVK) and the Radeon 780M. Making the texture with
  `PixelFormatView` usage avoids it.
- **A cold pipeline making on NVIDIA grows with a module's sampling sites.** NVIDIA's compiler
  (615.71.09) takes about 1 s for macOS 26's largest CoreAnimation fragment (210,033 words) on the
  RTX 5080 Laptop GPU, made once per cold cache (the first boot, and after an update changes the
  cache key); states that share the module then take 13 to 16 ms, and a warm cache 2 ms. The time
  follows the number of sampling sites: the module inlines a few sampling helpers five or six times
  each. Keeping one copy of each helper would cut it; it is not built.
- **An intrinsic declaration from macOS 12's and 13's Metal compiler.** The consumer's macOS 12
  and 13.7.8 guests have Contacts' compute kernels refused with
  `InvalidInput(Intrinsic(SignatureMismatch))` at a bitstream position in a function body (blocks
  `[8, 12]`). The consumer reports guests 11, 14, 15 and 26 clean (unverified here). Decode raises
  it (`intrinsics::bind`, from the call decode in `bitcode/function.rs`) for an `air.*` or `llvm.*`
  declaration whose own function type fits none of the operand forms `intrinsics::classify`
  derives from its name (`DeclClass::Mismatch`). It is not a call at a type other than its
  declaration's: decode binds a call to an intrinsic only when the two types are equal (pointee
  types erased), and `canon::post_check` refuses any other at a function location. The
  declaration and its form are unverified: no recorded translation of the private corpus (14,787
  rows, AIR 2.5 from Metal compiler 31001.723 among them) is refused this way. A fix needs the
  guest's library bytes to name the declaration, then that form as an `alternative()` of its family
  (as the legacy texture read and the AIR 2.7 and 2.8 forms are), decoded to the same operation and
  tested with a synthetic fixture; a declaration whose operands mean something else stays refused.
  It is not built.
- **A loop that Apple's MSL compiler folds into 33-bit arithmetic fails to make on MoltenVK.**
  Apple's MSL front end (Metal compiler 32023.921) can rewrite a loop that sums its counter
  (`1 + 2 + ... + n`) into a closed form computed in 33-bit integers (`zext i32` to `i33`,
  `mul i33`, `lshr i33`). Where that form lands in the kernel's constant program, the M3 Max's back
  end (`applegpu_g15s`) cannot legalize the 33-bit zero extension ("unable to legalize
  instruction ... (s33)", in `agc.main.constant_program`). Metal's compiler service then ends,
  and MoltenVK fails the pipeline's making (`VK_ERROR_INITIALIZATION_FAILED`,
  `XPC_ERROR_CONNECTION_INTERRUPTED`). The runtime returns that failure: a dispatch's command
  buffer completes with `CommandBufferError::PipelineCreation`, and an indirect command's commit
  returns `Error::Vulkan`. Hand-written MSL of the same loop fails the same way on Metal. The AIR
  Metal compiles directly gets no such rewrite, and runs. Measured on the committed case
  `authored-four-iteration-loop`: offline with `xcrun metal` and `metal-tt -arch applegpu_g15s`,
  and in the runtime lane on an M3 Max with macOS 26.7. No form of the loop is known that avoids
  the rewrite for every such loop, so it is not built.
