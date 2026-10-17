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
| A two-plane YCbCr texture's linear sample (`YCBCR8_420_2P` and its kin, over a surface's planes) is the device's filter of its texels, which hold Metal's reads exactly, as any other texture's is under native sampling: a draw binding one takes the pipeline a draw binding any other texture takes. Metal sums the texels' steps on the format's grid (`1/2048`, `1/8192`), each term truncated. Under native sampling, MoltenVK's included (Metal's own sampler filters the texels as an `RGBA16Float` or `RGBA32Float` texture's); with `sampling = SamplingPolicy::Exact` every module answers Metal's filter, MoltenVK's through the slot's variant. | `ycbcr_filter = YCbCrFilterPolicy::Device` | `ycbcr_filter = YCbCrFilterPolicy::Exact`: Metal's filter emulated in the slot's YCbCr variant, a pipeline the first draw binding such a texture makes | Linear filter within 1.183 codes (1/255) of Metal's filter of the format on NVIDIA (RTX 3090), 0.877 on RADV, 1.058 on ANV, 0.872 on lavapipe; within 0.498 at coordinates on the 1/256 grid. `DeviceSampling` (`device-sampling`). On MoltenVK within one step of the grid (0.125 codes on `1/2048`, 0.016 on `1/8192`), a `half` result within one `half` ulp; a `half` sample of one texel on the grid `8192` through a runtime sampler rounds to nearest even where Metal rounds ties away. `ycbcr-filter-not-exact`, `HalfTexelConversion` (`half-texel-conversion`), announced where a draw binds such a texture | One pipeline more, made at the first draw binding such a texture. A2X (`a2x-s96` with a `YCBCR8_420_2P` texture at fragment slot 3, on ANV, Iris Xe): 2 pipelines, 13.9 s at the state's creation and 37.8 s at the draw (52.3 s wall), against the creation's one drawn as made (14.3 s) |
| A texel's `float` value is the device's conversion of its codes; a `half` result is the device's `float` texel rounded to nearest even. | `texel_conversion = TexelConversionPolicy::Device` | `texel_conversion = TexelConversionPolicy::Exact`: Metal's conversion by the bound format's class of every read; a sample's or gather's also with `sampling = SamplingPolicy::Exact` (under native sampling the device's sampler decodes it) | One `f32` ulp for 8- and 16-bit codes, 0.18 of a code for 5- and 6-bit packed channels; `half` within one ulp; NaN bits the device's. `DeviceTexelConversion` (`device-texel-conversion`), `HalfTexelConversion` (`half-texel-conversion`, also recorded for heap textures and unmeasured formats) | Exact reads (the same reads): Xgc14 0.88-0.89 → 0.90-0.93 s, GPU 0.039 → 0.046 ms; Xgc15 0.65-0.66 → 0.68 s, 0.090 → 0.094 ms; A2X unchanged (no affected read). Samples: with the sampling row |
| The maths library (`sin`, `exp`, `log`, `pow`, `atan2`...), divisions and Metal's fast reciprocal, and a fused multiply-add without `OpFmaKHR` are the device's instructions, not routines of Metal's answers. Off MoltenVK a `float` multiply, and a fused multiply-add by `OpFmaKHR`, keep the device's rounding before Metal's flush: Metal flushes a result whose exact value lies below `2^-126` before rounding it. | `float_maths = FloatMathsPolicy::Native` | `float_maths = FloatMathsPolicy::Exact`; there off MoltenVK a `float` product of `±2^-126` is Metal's zero where the exact product lies below `2^-126`, decided inline without a branch by one `OpFmaKHR` on the operands' magnitudes, or by their bits without `OpFmaKHR` (`arith.float.tiny`, `dot`'s first product too), and `OpFmaKHR` is checked: a result at most `2^-126` in magnitude, zero included, is answered by a routine called only then, with `double` one that decides only the exact sum's sign and whether it lies below `2^-126`, else the integer `fma`; a module with either check, or with a fused multiply-add the translation computes itself (the `double` and integer routines without `OpFmaKHR`, and every `half` one), declares `SignedZeroInfNanPreserve` at 32 bits, and at 64 for a `double` routine, where the device supports it, so the device keeps the zero signs they answer (without it Vulkan lets the device answer a zero of either sign; else `DeviceSignedZeroInfNan`) | Vulkan's precision table (`Sin` 2^-11 absolute in [-π, π], `Exp` 3 + 2·\|x\| ulp, `Atan` 4096 ulp) against Metal's 4-6 ulp; division 2.5 ulp; `fma` one `half` ulp (a `float` one may round twice); an `f32` denormal in a maths function, `fast::fmod` among them, read as the device reads it (`DeviceDenormals`); a product or fused multiply-add whose exact value lies within half a unit below `2^-126` answers `2^-126` of its sign where Metal answers the zero of its sign, announced once per device as it is made. `NativeTranscendentals` (`native-transcendentals`), `NativeDivision` (`native-division`), `NativeFma` (`native-fma`), `FloatTininessAfterRounding` (`float-tininess-after-rounding`) | A2X 0.99-1.02 → 9.66-9.83 s, GPU 0.097-0.098 → 0.222-0.225 ms; Xgc14 0.87-0.90 → 12.6-13.6 s, 0.038 → 0.124-0.135 ms; Xgc15 0.63-0.65 → 9.40-9.54 s, 0.089 → 0.208 ms (at 7a9d65147f, before these modules declared `SignedZeroInfNanPreserve`; its cost is not measured). Off MoltenVK a product's check is 12 instructions inline (23 without `OpFmaKHR`), and a checked `OpFmaKHR` adds its test, a branch and a call of a routine of about 40 instructions (with `double`), which the device's compiler copies into each use |
| A NaN result's bits are the device's (no select of Metal's NaN), and `fmin`, `fmax`, `min3`, `max3`, `clamp` and `saturate` are the device's `NMin`, `NMax` and `NClamp`. | `nan_and_min_max = NanMinMaxPolicy::Native` | `nan_and_min_max = NanMinMaxPolicy::Exact` | A NaN where Metal answers one, its bits the device's; the operand Metal picks but at two NaNs, `+0` against `-0`, and `f32` denormals. `NativeNanEncoding` (`native-nan-encoding`), `NativeMinMax` (`native-min-max`) | A2X 1.00-1.02 → 1.85-1.88 s, GPU 0.098 → 0.116 ms; Xgc14 0.88-0.89 → 1.59-1.63 s, 0.039 → 0.048 ms; Xgc15 0.65-0.66 → 1.25-1.28 s, 0.090 → 0.102 ms |
| `dot`, `mix` and `fast::sqrt` are the device's `OpDot`, `FMix` and `Sqrt`, not Metal's fused chain, its two fused multiply-adds and `x · rsqrt(x)`; a `dot` of other than two to four lanes sums the `OpDot` of each chunk of up to four lanes. | `rounding_order = RoundingOrderPolicy::Native` | `rounding_order = RoundingOrderPolicy::Exact` (MoltenVK takes Metal's order under every policy): Metal's chain over every lane of a `dot` of any width (test `wide_dot_is_metals_fused_chain_in_metals_order`) | `float` `dot` of n lanes within (3n − 1)·2^-24·Σ\|aᵢbᵢ\|, `mix` within 2^-21 of its two terms, `fast::sqrt` 4.5 ulp; `FMix` and `Sqrt` read an `f32` denormal as the device does (`DeviceDenormals`). `NativeDot` (`native-dot`), `NativeMix` (`native-mix`), `NativeTranscendentals` (`native-transcendentals`); a `dot` of other than two to four lanes within Vulkan's precision of its chunks' `OpDot` and sum, an `f32` denormal read as the device reads it, `DeviceArithmetic` (`device-arithmetic`) and `DeviceDenormals` | A2X 1.00-1.02 → 1.03-1.06 s, GPU 0.098 → 0.101 ms; Xgc14 0.88-0.89 → 0.97-1.02 s, 0.039 → 0.041 ms; Xgc15 0.65-0.66 → 0.73 s, 0.090 → 0.088 ms |
| A `half` result computed in `float` rounds through the device's conversion to 16 bits (three instructions), or its `OpQuantizeToF16` without `shaderFloat16`, not Metal's rounding in integer arithmetic (about twenty), on a device whose conversion is neither measured to be Metal's nor made Metal's by its float controls. ANV and RADV round through the conversion and its widening, each `NoContraction`, under `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits: exact by the Vulkan spec under either policy, no note. NVIDIA's driver folds that pair (`DriverWorkaround::HalfConversionPairFolded`), so NVIDIA takes this row. | `half_rounding = HalfRoundingPolicy::Native` | `half_rounding = HalfRoundingPolicy::Exact` | One `half` ulp; 65504 where Metal answers infinity on a device rounding toward zero; a half denormal may flush; NaN bits and a zero's sign the device's. Or no rounding at all, measured: NVIDIA (RTX 3090, 615.71.09) drops the conversion pair, and lavapipe does in arithmetic on Mesa 26.2.4 with LLVM 23.1.1 and for a loaded texel on Mesa 26.0.8 with LLVM 21.1.8, whose arithmetic measured rounded, so the result keeps `float` precision, its differences compounding across operations, and a value past `half`'s range stays finite until a `half` store saturates it to infinity; the RTX 5080 Laptop GPU (615.71.09) rounds, measured: 8192 of 8192 narrowed sums are Metal's. `NativeHalfConversion` (`native-half-conversion`), `NativeHalfQuantize` (`native-half-quantize`) | A2X 0.99-1.04 → 7.19-7.21 s, GPU 0.098-0.100 → 0.139-0.140 ms (+40%); Xgc14 0.86-0.87 → 0.87-0.92 s, Xgc15 0.63-0.66 → 0.69-0.70 s, GPU unchanged (RTX 5080). Rounding natively by `OpQuantizeToF16` instead, which the RTX 3090 and lavapipe honour (Metal's answer on every operand tested), was measured and declined: A2X 1.3 s → 18-22 s, GPU 0.14 → 1.5 ms on an RTX 3090 |
| A `half` read from its bits (a load, a reinterpretation, a texel's bits) outside every loop is widened by the device (`UnpackHalf2x16`), not decoded exactly. | `half_rounding = HalfRoundingPolicy::Native` | `half_rounding = HalfRoundingPolicy::Exact` (with the row above) | A half denormal may read as zero where the device flushes them; NaN payload the device's. `NativeHalfConversion` (`native-half-conversion`) | Not measured alone on today's master. At its merge: A2X 1.04-1.07 s with it against 1.27-1.28 s with the exact decode; Xgc unchanged |
| The `half` conversion converts the value itself (no `NoContraction` barrier), so the device's compiler may fold it into the instruction computing the value, rounding once. | `half_rounding = HalfRoundingPolicy::Native` | `half_rounding = HalfRoundingPolicy::Exact` (with the rows above) | The row above's: on NVIDIA (RTX 3090), and on lavapipe for a loaded texel (Mesa 26.0.8), the pair is dropped whether or not the barrier stands, so the barrier does not restore the rounding there. `NativeHalfConversion` (`native-half-conversion`) | At its merge: A2X 0.99-1.02 s without the barrier against 1.00-1.05 s with it; Xgc within a round's spread |
| A `half` texel's NaN is the rounding's own NaN, not the texel's bits (the keep's four instructions a texel are left out). | `half_rounding = HalfRoundingPolicy::Native` and `nan_and_min_max = NanMinMaxPolicy::Native` | `nan_and_min_max = NanMinMaxPolicy::Exact` (the keep under native half rounding, master ecd39568bb); `half_rounding = HalfRoundingPolicy::Exact` also keeps it, at that row's cost. The same switch also gives Metal's full rule: `0x7e00` for every NaN but a one-texel read, nearest sample or gather of a 16-bit float texture, by the slot's format class read as data, a heap handle's class index and a runtime sampler's filter. A gather with a `half` result fetches its four texels where the class is a half-float one, since NVIDIA's gather hands back every NaN as `0x7fffffff`. **Not yet exact:** a cube gather, bicubic and emulated linear, fetched or compared samples; and on ANV (Iris Xe, Mesa 26.2.4) every 16-bit float texel's NaN, which its image decode answers as `0xFFC00000` (`0xFE00` as a `half`) before any keep: reading the texel's bits through an unsigned integer view would answer Metal's. Possible, not built: the runtime sampler's filter read as push data rather than specialized, which would spare the remake below. Measured on the RTX 5080: under the switch no `half` lane of the runtime lane's texel reads, samples and gathers differs from Metal's (1,469 differ by default). Under the switch `DeviceNanEncoding` is still announced; the consumer logs notes, so no device fact is built to drop it | NaN bits the device's (`0x7FFF` on the RTX 5080), where Metal hands back a 16-bit float texel's own bits on a read, nearest sample or gather, and `0x7e00` for a 32-bit or small float texel's NaN and a linear sample's. `NativeHalfConversion` (`native-half-conversion`); the read also records `DeviceNanEncoding` (`device-nan-encoding`) | Measured on top of the NaN switch's own row (its probe with the keep, c0a814a13, against its probe without, 3e8a01c54): A2X 341,299 → 347,519 words, 1.82-1.86 → 1.81-1.87 s, GPU 0.115-0.117 → 0.115-0.116 ms, within noise; Xgc14 and Xgc15 read no `half` texel (words unchanged). The full rule (its probe a5c642ebe against the same tip without it, 851650c81): A2X 347,519 → 372,429 words (+7.2%), creation 1.85-1.88 → 1.99-2.04 s; where a linear runtime sampler is bound at the draw the sampler word's filter is now specialized, so the first draw remakes (1.94-1.96 s) and its GPU time rises 0.115-0.117 → 0.124-0.125 ms; with a nearest sampler GPU 0.115-0.116 → 0.115-0.117 ms; Xgc14 and Xgc15 words unchanged. On the RTX 3090 the same pair: A2X creation 2.91-3.06 → 3.17-3.23 s, the first draw's remake 3.0 s and its GPU time 0.17-0.18 → 0.79-1.15 ms, with a nearest sampler 0.17-0.18 ms both |
| On ANV and RADV, which round a `half` through the device's conversion under the float controls, a `half` read from its bits (a load, a reinterpretation, a texel's bits) is the device's widening of them as a 16-bit float, loops included, and a `half` texel takes that conversion's NaN (no keep): every value but a NaN is exact by the Vulkan spec. | `nan_and_min_max = NanMinMaxPolicy::Native` | `nan_and_min_max = NanMinMaxPolicy::Exact`: the integer decode and the texel's keep; a `half` widened to `float` then answers Metal's `0x7fc00000` for every NaN, read from memory, a texel or a select alike | A `half` NaN widened to `float` has the device's bits (ANV quiets a signalling NaN, RADV keeps it), where Metal answers `0x7fc00000`; a `half` stored back unchanged keeps its bits. `NativeNanEncoding` (`native-nan-encoding`); a texel's, `DeviceNanEncoding` (`device-nan-encoding`) | ANV (Iris Xe, Mesa 26.2.4), A2X: making 4.06-4.36 → 5.76-5.87 s, 269,828 → 280,070 words, peak RSS 0.35 → 0.51 GB, GPU 4.48 → 4.62 ms (warm); RADV (Radeon 780M, Mesa 26.2.4): 379-467 → 535-665 ms, GPU within its clock states' spread |
| A fragment function without fast math returns a `half` color to a blended attachment: the device blends in its own arithmetic and rounds once, where Metal blends a 16-bit float attachment in `half` arithmetic (each factor, product and sum rounded to `half`). | `half_blend = HalfBlendPolicy::Device` | `half_blend = HalfBlendPolicy::Exact`: the function computes Metal's blend in the interlocked storage form, for a `half` output of one to four lanes into `R16Float`, `RG16Float` or `RGBA16Float` where that form is exact for the pipeline (RADV, ANV, NVIDIA, MoltenVK); elsewhere, and for lavapipe, 32-bit float, 8-bit, a module naming no math mode or a blend of the second source, the device's blend and its note stay | 2.5 `half` ulps at the largest operand on NVIDIA (the blend color and one minus it), two on RADV, ANV and lavapipe (16-bit float); two `float` ulps (32-bit float, one on NVIDIA); one or two codes (8-bit, extended range); MoltenVK stores Metal's bytes. Measured with and without the factors' magnitudes at 0bcc0163e5. `HalfOutputPrecision` (`half-output-precision`) | A probe at 924fc11603 (a `half4` over a 1920x1080 `RGBA16Float` target, One / OneMinusSourceAlpha): making 11.1 → 24.0 ms; GPU 25 → 82 us one draw, 506 → 2291 us 32 overlapping draws (the interlock orders them). RTX 3090 (the same draw at 50a065eb98): making 12.9 → 27.4 ms; GPU 45 → 115 us one draw, 648 → 3268 us 32 overlapping draws |
| A texture over a buffer whose rows (or an array's slices) lie with bytes between them, and which only reads (its usage, as Metal reports it, holds no `RenderTarget`, `ShaderWrite`, `ShaderAtomic` or block writes; `Unknown` holds them all), is an image over the buffer's bytes at Metal's row pitch, as one of packed rows is: the driver's own linear image, a `DRM_FORMAT_MOD_LINEAR` image at that pitch, or one over a dma-buf of the pages. Its subresources hold the bytes between its rows, which Metal never writes, and its first move reaches them. The runtime's choice: no pipeline differs. | `padded_reads = PaddedReadsPolicy::Alias` | `padded_reads = PaddedReadsPolicy::Shadow`: the runtime's shadow image of the texels, copied in from the buffer before each use (`shadowed-texture`), whose copies address the texels alone | The host keeps the bytes the image is bound to, the bytes between the rows among them, across its first move (the known gap below): a CPU write to them within one submission's round trip of the commit that makes the move is undone, once per image, unannounced, where Metal keeps it; no byte outside them is touched. `runtime_lane::pitched_surfaces::a_texture_that_only_reads_keeps_every_byte_of_its_pages` fails on a device whose move changes one past the keep | ANV (Iris Xe; `surface_texture_costs::read_only_padded_costs`), a frame sampling four `BGRA8Unorm` tiles over no-copy pages: textures made once, aliases and shadows about level (3.0-6.1 against 3.2-5.3 ms); a new texture over the pages each frame, aliases 4.6-8.6 ms (their first moves kept on the host) against shadows 3.4-7.1 ms |
| A computed `half` or `float` color blended into an extended-range attachment (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`) rounds the blend's source as a loaded one; Metal's compiler rounds it by how it computed it. Every device. | The in-shader blend's loaded rounding (no switch) | **Not built yet:** Metal's rounding per shape is probed, not modelled | One code. `ExtendedRangeBlendSource` (`extended-range-blend-source`), `ExtendedRangeFloatBlendSource` (`extended-range-float-blend-source`) | Not measured |
| A bicubic sample of an 8- or 16-bit normalized texture sums its taps exactly and rounds half up; Metal truncates its row terms, row sums and column terms first. Every device. | The exact sum (no switch) | **Not built yet:** Metal's datapath is pinned (17.75M lanes measured, no miss); its emitter is not written | One grid step (1/4080 at 8 bits) within about 2^-21 of a tie. `BicubicFilterArithmetic` (`bicubic-filter-arithmetic`) | Not measured (Metal's form is expected no larger) |
| A bicubic sample of an extended-range texture (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their sRGB twins) sums its taps' steps exactly, rounds half up and clamps into the format's decoded endpoints; within 0.014 of a grid step of a rounding tie Metal may round the other way. Every device. | The exact sum (no switch) | **None known:** no model of Metal's datapath tried fits; 3,317,760 lanes measured on the M3 Max (macOS 26.7) | One grid step (1/8160 a color lane, 1/8192 an sRGB color lane, 1/4080 alpha), 4,050 of the lanes measured, each within 0.014 of a step of a tie. `BicubicFilterArithmetic` (`bicubic-filter-arithmetic`) | No accurate form |

All at once: `DevicePolicy::exact_answers()` sets every switch above but sampling's (validation
adds `sampling = SamplingPolicy::Exact`). Against master 1ee783c77 it cost A2X 0.99-1.04 → 17.5 s,
GPU 0.098 → 0.267 ms; Xgc14 0.86-0.87 → 4.95-5.01 s, 0.039 → 0.105-0.110 ms; Xgc15 0.63-0.66 →
3.06-3.15 s, 0.090 → 0.194-0.197 ms.

Speed choices that change no answer (the roll pass's floor; a specialization measured free) are not
rows.

## Not speed choices: device gaps

No switch gives Metal's answer here: the device lacks what an exact form needs, or no exact form is
known. The rest of the differences are docs/API.md section 9's precision notes.

| What it is | Why there is no switch | Note, code |
|---|---|---|
| `f32` denormals in the operations left to the device under every policy: a conversion to or from `double`. Metal reads them as zero. | Vulkan lets each device keep or flush them; Metal runs no `double`, so its reading of a conversion is not measured. Native device maths and the device's order record the note too (their rows above). Every other `float` operation reads them as Metal on every device (test `the_accurate_forms_read_denormals_as_metal`). | `DeviceDenormals` (`device-denormals`) |
| A denormal 32-bit float texel the device's sample or gather reads as zero (RADV's gathers and nearest samples, NVIDIA's gathers, ANV's nearest samples), where the module cannot fetch the texels itself. | The device's filter decides; no fetch form reaches filtered, cube, heap or runtime-sampler samples. | `DeviceTexelDenormals` (`device-texel-denormals`) |
| Depth values outside [0, 1] stored clamped (RADV, ANV) on a device without `VK_EXT_depth_range_unrestricted`. | Vulkan leaves them undefined without the extension: enable it. | `RestrictedDepthRange` (`restricted-depth-range`) |
| Signed zeros, infinities and NaNs where the device lacks `SignedZeroInfNanPreserve` at the operation's width. | A device capability. | `DeviceSignedZeroInfNan` (`device-signed-zero-inf-nan`) |
| Every device but MoltenVK interpolates varyings by its own rounding, up to about `1.5e-4` from Metal's (`3e-4` on lavapipe). MoltenVK hands the varyings to Metal. | The exact path needs Metal's interpolation formula, which is not known. | `DeviceInterpolation` (`device-interpolation`) |
| A fast-math function's `half` color into a blend: within three `half` ulps (16-bit float) or two `float` ulps (32-bit float). A fetched 16-bit float attachment's blend, computed in the pixel interlock, where Metal folds the output's own arithmetic into it or regroups it: within 2 `half` ulps at the operands' magnitude, 16 at the stored value's under cancellation; exact elsewhere. | Metal's compiler regroups the blend per pipeline, following its passes over the whole function; no form the function and the blend state decide is Metal's. | `HalfOutputFastMathBlend` (`half-output-fast-math-blend`) |
| Float simdgroup reductions sum in the device's association order. | Apple's order is not modelled. | `ReductionOrder` (`reduction-order`) |
| A vertex or fragment function keeps more than 16 KiB a thread of private memory, up to the 192 KiB Metal holds, on a device whose Function storage was not measured: it runs on the driver's scratch, which past what the driver holds answers wrongly or hangs the GPU with no error. | Vulkan states no amount of Function storage; only a measurement of the device does. No other form is exact: a fragment function's helper invocations may not write the memory it would move to, and a vertex function's, a compute pre-pass, is not built. | `FunctionStorageAssumed` (`function-storage-assumed`) |
| A render pipeline of more color attachments than the device draws at once (the lesser of `maxColorAttachments` and `maxFragmentOutputAttachments`, four at Vulkan's floor; Metal takes eight) draws each draw once per group of that many, each group in a render pass of its own over a twin of the pass's depth and stencil. Every variant's vertex stage is one module, `Position` invariant, and every fragment variant's `FragDepth`, `SampleMask` and `FragStencilRefEXT` are `Invariant`; groups past the first may differ at discard or depth edges if the driver compiles the variants differently. | The device holds no more attachments in one subpass. Vulkan's invariance rules make only identical pipelines repeat exactly, and the groups' fragment modules and attachments differ. | `ColorAttachmentGroups` (`color-attachment-groups`) |
| Command buffers that share no resource run one after another, where Metal overlaps them while the GPU has room. | The runtime orders nothing between them and runs at each driver's floor for one completion signal per command buffer; a signal shared by several would hold an earlier one's completion for the later ones' work. | None (timing only) |
| On MoltenVK a light queue's command buffer waits for the latest one another queue committed, where Metal waits for none: MoltenVK fences every encoder of every queue device-wide when argument buffers and a residency set are on. | The runtime needs argument buffers, and MoltenVK has no setting for its residency set; `runtime_lane::queue_storms` judges the runtime against Metal under those fences. | None (timing only) |
| A `Counting` visibility result of a draw writing more than one color attachment, on ANV (Mesa 26.2.4): Metal's count multiplied by the colors written, not divided back. `Boolean` results stay exact. | The driver counts each sample once per color output (`RuntimeDriverDefect::OcclusionCountPerColorOutput`), no documented Vulkan form counts it once, and a refusal would withhold the Mac2 and Metal3 families. | `VisibilityCountPerColor` (`visibility-count-per-color`) |
| On MoltenVK a no-copy buffer or heap released with no keep-alive while batches are in flight runs its deallocator once the batches submitted before its release completed, where Metal runs it at the release. | MoltenVK 1.4.1 fails pending work that reads device addresses where memory a device-address buffer lay in is freed (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), so freed memory waits for the batches in flight (`WorkCounts::freed_memory_parked`). | `HostDeallocatorDelayed` (`host-deallocator-delayed`) |
| On ANV (Mesa 26.2.4) a render pipeline whose largest stage module, under the consumer's exact forms, passes 400,000 words is retranslated with those forms dropped until it fits: native sampling first, then every module-growing exact answer native. On a large enough module the driver's register allocator reads past its interference bitset and the process faults at pipeline creation. | ANV's `brw` register allocator faults on such modules (`DriverWorkaround::RegisterAllocatorOverflowsLargeModules`), no documented Vulkan form bounds a module's register pressure, and a refusal would withhold the consumer's large fragments. | `ExactFormsOverCompilerLimit` (`exact-forms-over-compiler-limit`) |

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

## Costs of matching Metal

Where matching what Metal observably does costs time, the runtime matches it and the cost is
recorded here.

- **`commit` returns at once.** Metal's `commit` returns before its command buffer is scheduled.
  The runtime's commit hands the batches to the device's scheduler thread, which submits them
  (`scheduler/inbox.rs`), so no driver call holds `commit`. Where that thread sleeps, a commit's
  batches wait for its wake: three command buffers committed back to back and waited for take a
  median 90 to 160 µs longer on ANV (Iris Xe), 5 µs on the RTX 5080 Laptop GPU and none on the M3
  Max. On the Iris Xe, frames of 24 command buffers committed back to back
  (`runtime_lane::commit_costs`, `frames`) take a median 5.7 to 6.0 ms against 3.2 to 4.5 ms with
  the submissions on the committing thread, the GT held at its efficient 300 MHz by its governor
  (GuC SLPC). A submission check or submission handler, which run on that thread, holds back every
  other command buffer's submission while it runs.
- **Idle dma-buf imports are released.** RADV lists every imported memory in every kernel
  submission, so a no-copy buffer's dma-buf import that only blits have read is released after its
  idle window (2 s, doubled at each import again up to 64 s). The encoding thread pays its import
  again at the first blit after the window (HOWTO.md section 6).
- **A pass in color groups runs once per group** (the `ColorAttachmentGroups` row above), the later
  groups' draws again over a twin of the depth and stencil, made for the pass and loaded by a copy
  or a clear (`render/color_groups.rs`). On lavapipe restricted to four colors
  (`runtime_lane::color_groups::color_group_costs`), 16 fullscreen draws into five 512²
  `RGBA16Float` colors took 1.04 times the same draws into four from commit to completion, and 1.51
  times with a loaded `Depth32Float_Stencil8` the draws write.
- **A draw that reads its own pass's attachment reads a copy, a snapshot or a split.** A sampled
  slot binding of a texture the render pass instance renders reads a copy taken before the
  instance (`texture/attached_reads.rs`): the texels around the draw's scissors where it reads at
  its own pixels alone, else every texel or what the instances since drew. After a barrier inside
  the instance, an own-pixel read takes a snapshot drawn in its subpass (`render/snapshot.rs`,
  docs/API.md "Snapshot reads") or splits the instance; both are exact. The rule weighs the
  snapshot draw at 20 bytes of the split's copy for each pixel its scissors bound, which fits the
  Iris Xe, the RX 7900 XTX and the RTX 5080 Laptop GPU; on the Radeon 780M the split is faster from
  about 14.5%, 17% and 42% of the target for 4-, 8- and 16-byte texels, so the rule costs up to 36%
  there (`RGBA16Float` at 30% of a 1920x1080 target).
- **A `Depth16Unorm` depth whose code the fragment stage writes is tested after the stage.** Where
  the device's conversion is not measured to store Metal's code
  (`DriverDeviation::MetalDepth16Stores`, or `MetalDepth16BiasedStores` for a biased draw), a
  single-sampled draw writes each depth as Metal's code from its fragment stage
  (`ShaderUnormDepth`), so a fragment the depth hides still runs it. A bias of a known side
  declares it (`DepthGreater` or `DepthLess`) and the device may test first; an unbiased draw and
  a bias whose side a triangle's slope decides keep the cost. On the Iris Xe 16 full-screen layers
  drawn front to back under a constant bias took 9.42 ms of GPU time with no side declared and
  1.85 ms with one, against 1.46 ms unbiased through the device's conversion
  (`runtime_lane::unorm_depth_bias`).
- **A stored negative or NaN depth is tested by its magnitude.** A switched-on single-sample
  `Depth32Float` texture's pass that loads its depth takes two copies through a scratch buffer and
  a dispatch over the render area before it, and the same after it where it writes the depth; a
  sign strip (two copies and a dispatch, or a draw per layer when multisampled) follows a pass with
  depth writes where no stand-in runs (`texture/signed_depth.rs`). A texture never switched on pays
  a flag read per pass.
- **A texture over a buffer whose rows lie with bytes between them, which renders or which a shader
  writes**, is a shadow on every device but MoltenVK (`shadowed-texture`): its texels copied in
  before each use and back after each write (only those inside the draws' scissors for a pass that
  takes it as a loaded attachment alone): an image over those rows holds the bytes between them in
  its subresources, which a render pass's load and store operations and the image's first move may
  rewrite. On ANV making a 1307x512 tile took about 135 µs as a shadow against 41 µs as an image.
  Packed rows avoid it.
- **On MoltenVK, a texture over a buffer that renders or that blits write keeps a shadow.** MoltenVK
  1.4.1 reports a format's linear-tiling attachment and blit-destination features from
  uninitialized memory, so no MoltenVK description states them
  (`DriverFact::LinearTilingFeaturesUninitialized`) and such a texture is a shadow
  (`shadowed-texture`).
- **A view of another format without `PixelFormatView` usage** is a shadow image copied at each use
  (`shadowed-texture-view`): about 50 to 95 µs a command buffer for a 64x64 texture. Making the
  texture with `PixelFormatView` usage avoids it. A view in other channels of its image's bits (an
  `A8Unorm` view of an `R8Unorm` texture) binds the image itself through a swizzle, and only a pass
  that renders into it renders through a shadow.
- **A pass that loads a texture over a buffer and does not store it** renders through an image its
  texture keeps, the buffer's texels copied in first (`dont-care-stand-in`), so the buffer keeps its
  bytes as Metal's does: about 10 µs a pass on the Iris Xe (ANV). A pass that clears and stores
  `DontCare` costs no more, and one that stores renders in place.
- **The interlocked pass on the M4 under macOS 27.** MoltenVK gives the M3 Max and the M4 one device
  identity on macOS 27, so both split a pass the pixel interlock would split
  (`DriverFact::InterlockedPassSplitNoSlower`), about a quarter faster on the M3 Max; the M4 runs
  that pass about 8% faster kept whole.
- **Emulated tessellation's scan.** On a device without `tessellationShader` a patch draw first
  runs a one-workgroup scan that places each patch's triangles in Metal's order, patch after
  patch: about 6-60 ms of GPU time per 10,000-patch draw at factors 4 to 64 on the Radeon 780M and
  the RTX 3090, 0.6-5 ms on lavapipe (`emulated-tessellation-draws`). A parallel form is not built.

## Known gaps

Open differences from Metal that no note announces yet, and runtime costs still above the
driver's. Each is open work.

- **A mesh or tile render pipeline past ANV's compiler word bound.** Its stages are not fitted to
  the bound as a vertex and fragment pipeline's are, so under the exact forms ANV's compiler can
  fault on it.
- **A level the device's arithmetic chooses.** Off MoltenVK under exact sampling, a fragment's
  implicit-level sample (its bias Metal's `B` within the device's limit), a sample with explicit
  gradients and a fragment's LOD query take the device's level of detail, where Metal's is not
  measured, and an explicit level Metal's rule is not handed on (an integer texture's, a constexpr
  sampler's with its own bias or an unmeasured clamp) takes the device's rounding. Where the level
  chooses what a sample reads and no sample-filter note records it, and for every LOD query, it is
  announced as `DeviceSampleLevel`.
- **An anisotropic cube sample on a `y` face through MoltenVK's vertex and fragment stages under
  macOS 27.** SPIRV-Cross's `spvGradientCube`, which MoltenVK 1.4.1 applies, writes a `+Y` or `-Y`
  face's gradients as `(d_x, d_z, d_x)`, so Metal loses the `z` gradient. The module hands such a
  face Metal's length of the two tangents on both (`CubeGradientRoute::AppleProjected`): isotropic
  samples, and every sample on the `x` and `z` faces, are Metal's, but an anisotropic one on a `y`
  face filters along the face's diagonal, announced as `CubeGradientFootprint`
  (`cube-gradient-footprint`). A kernel's samples are Metal's own. A form exact inside the face,
  sampling a `y` face through a 2D-array view of the cube, needs a second binding of each such cube
  from the binder and the runtime; it is not built.
- **An anisotropic cube sample out of the shader's reach through MoltenVK where its cube-gradient
  route is not measured** (`CubeGradientRoute::Unmeasured`). The shader takes Metal's level and taps
  wherever its emulation reaches (a cube or cube array of `float` or `half` texels, without
  comparison, through a constexpr sampler or a runtime sampler slot under its variant). A depth or
  comparison sample, one whose residency the function reads, an argument buffer's texture or
  sampler, or a slot of another class or swizzled storage takes Metal's isotropic footprint through
  a sampler made isotropic for that use, where Metal filters an ellipse, announced as
  `CubeGradientFootprint`. An argument buffer's sampler needs its degree in its table entry; it is
  not built.
- **A group operation in a branch arm whose speculation no probe covers.** Metal's backend runs a
  branch's short arms of plain computation in every lane of the branch's block; the module does the
  same for the arm shapes measured on Apple's GPU (`air::canon::speculate`). A branch of another
  shape whose arms hold a quad or simdgroup operation keeps the AIR's lanes, where Metal's may be
  the block's, announced as `UnmeasuredGroupSpeculation`.
- **A render pass's attachment read another way than a sampled slot binding.** Vulkan forbids
  reading an attachment in its own render pass instance (`VUID-vkCmdDraw-None-10795`); a sampled
  slot binding reads a copy or a snapshot (Costs above). Every other such read still reads the
  attachment in its instance, announced as `AttachmentReadInPlace`: a texture reached through the
  heap set (a handle, or an argument buffer's texture), a storage binding, a depth or stencil
  attachment a draw samples, a texture buffer or a buffer over the same buffer's bytes, a texture
  whose shadow is not served (`texture/usage_shadow.rs`), and an execution's command
  (`render/executions.rs`). In an instance that holds the attachment in the feedback loop layout,
  those reads name `VK_IMAGE_LAYOUT_GENERAL` in their descriptors, against the layout the instance
  holds.
- **A stored negative or NaN depth where the magnitude stand-in and the sign strip do not reach.**
  Metal's depth test compares the stored value's magnitude (a NaN above +inf), and a pass with depth
  writes stores every texel no fragment wrote without its sign; the runtime matches both through a
  magnitude stand-in or a sign strip (`texture/signed_depth.rs`, docs/API.md's
  `signed-depth-test`). Still open:
  - a multisampled or `Depth32Float_Stencil8` target's test, which compares a negative value or a
    NaN as it is, and a pass with depth writes on a device that makes no strip for it, which keeps
    the signs: announced (`signed-depth-test`); its resolve in such a pass resolves the samples
    before the strip, unmeasured on Metal;
  - a merge that keeps a stored NaN where a draw wrote exactly +inf over it, and a denormal at
    exactly 2^-126 where the device's test flushes one: announced (`signed-depth-test`,
    `depth-denormals`);
  - a pass that loads the depth through another image, and one encoded before the switching write
    in another command buffer uncommitted then: announced (`signed-depth-test`);
  - a NaN clear depth where the device writes no `0x7fc00000` into the depth before the pass:
    cleared to 1.0, announced (`signed-depth-test`);
  - a device that may clamp a depth outside `[0, 1]` (ANV): announced as `restricted-depth-upload`;
    a form that keeps the value in a 32-bit float colour image beside the stand-in is not built.
- **Compute private memory past a device's Function storage is refused.** A kernel, object or mesh
  function whose private memory (its `alloca`s, a call stack among them, and constants held as
  `Private` variables) passes what the device holds is refused (`Unsupported{PrivateMemory}`, or
  `Unsupported{Recursion}` for a call stack): past 16 KiB a thread on a device not measured, past
  its measured value on a measured one. So is a vertex or fragment function past a measured
  device's value, or past 192 KiB elsewhere (below that it runs with `FunctionStorageAssumed`, the
  device-gap row above). Vulkan states no limit on the driver's scratch, so the lane
  `runtime_lane::private_memory` measures it per device and `device::function_storage` keys it on
  the driver, vendor and device ID: 96 KiB on the Radeon 780M and RX 7900 XTX, 32 KiB on the Iris
  Xe, the UHD 770 and the RTX 5080 Laptop GPU, 128 KiB on the RTX 3090, 192 KiB on the M3 Max and
  M4 under MoltenVK, 48 KiB on lavapipe; 16 KiB elsewhere. An exact form for compute, private memory
  in a runtime buffer with a slot per resident threadgroup, is not built.
- **A vertex function that selects its slice, on a device without `shaderOutputLayer` and
  `geometryShader`.** Metal renders each primitive into the slice its first vertex selects. Vulkan
  stores `Layer` from a vertex function only with `shaderOutputLayer`, and the generated geometry
  stages that store it otherwise need `geometryShader`; a device with neither refuses the pipeline,
  `Unsupported(LayeredRendering)` needing either feature (among the committed profiles
  `vk12-floor`, `roadmap-2022` and `lunarg-desktop-baseline-2023`). So does a vertex function that
  selects its viewport there with `multiViewport`. One whose slice or viewport differs among a
  primitive's vertices needs the stages even with the outputs (refused needing `geometryShader`
  without them), except on MoltenVK, which hands the indices to Metal. An exact form, a
  layered pass drawn once per layer keeping the primitives whose first vertex selects it, is not
  built.
- **Commit to completion.** A command buffer takes longer from commit to completion through the
  runtime than a bare driver submission: on a quiet Radeon 780M (RADV) an empty command buffer took
  51.6 µs against a bare barrier's 45.1 µs. On MoltenVK most of the rest is the two GPU timestamps
  each timed command buffer writes: on an M4 a bare barrier took 14.7 µs alone and 167.8 µs with
  them. `DeviceOptions::gpu_times_from_host` (off by default, announced) answers GPU times from the
  host clock and spares them.
- **A shadow's copy back after a pass that draws covers every texel inside the draws' scissors**,
  where Metal's store writes back only the 32x32 tiles a draw's primitives touch. A CPU write during
  the pass to an undrawn texel in an untouched tile inside the scissors can be undone where Metal
  keeps it: the host does not know which tiles the primitives touch. No copy reaches the bytes
  between rows.
- **A CPU write to a texture's bytes while the host keeps them across the image's first move is
  undone.** An image over a no-copy buffer's host pages starts `UNDEFINED`
  (`VUID-VkImageCreateInfo-pNext-01443`), and its first move may read and write all the memory
  bound to it, so the host keeps those bytes (`texture/first_move.rs`, `device/submit.rs`): once
  the submissions before have completed, it reads them through the host pointer, submits the moves
  alone, waits, and writes back those that differ. A CPU write to them in between, within one
  submission's round trip of the commit that makes the move, is undone, once per image
  (`texture/shared_alias.rs`); Metal keeps it, and never writes the bytes between a texture's rows.
  No byte outside them is read or written (`runtime_lane::pitched_surfaces`:
  `a_texture_s_first_move_keeps_the_bytes_it_lies_over`,
  `a_cpu_write_during_a_texture_s_first_move_stays_outside_its_bytes`).
- **A cold pipeline making on NVIDIA grows with a module's sampling sites.** NVIDIA's compiler takes
  about 1 s for macOS 26's largest CoreAnimation fragment (210,033 words) on the RTX 5080 Laptop
  GPU, once per cold cache; states that share the module then take 13 to 16 ms, and a warm cache
  2 ms. The module inlines a few sampling helpers five or six times each; keeping one copy of each
  would cut it. It is not built.
- **An intrinsic declaration from macOS 12's and 13's Metal compiler.** The consumer's macOS 12 and
  13.7.8 guests have Contacts' compute kernels refused with
  `InvalidInput(Intrinsic(SignatureMismatch))`: decode raises it (`intrinsics::bind`) for an
  `air.*` or `llvm.*` declaration whose type fits none of the forms `intrinsics::classify` derives
  from its name. AIR 2.5 (Metal compiler 31001.723) declares the indirect command setters
  `air.set_kernel_buffer_compute_command` and `air.set_vertex_buffer_render_command` without the
  attribute stride, which decode takes as MSL's plain setter; whether the Contacts kernels are
  those setters, and so translate, is unverified.
- **A direct call whose arguments do not line up with its callee's parameters** is refused with
  `InvalidInput(Canon(CallTypeMismatch))` (`canon::post_check`): LLVM leaves such a call undefined,
  and what Metal's back end makes of it is not documented, so no exact form is known. Four private
  corpus rows of one fragment function are refused so.
- **A loop that sums its counter fails to make through MoltenVK.** Apple's MSL front end rewrites a
  loop whose running sum adds its 32-bit counter (`1 + 2 + ... + n`) into a closed form in 33-bit
  integers, which the M3 Max's back end cannot legalize where it lands in the kernel's constant
  program: MoltenVK, whose SPIRV-Cross writes such a loop in that shape, fails the pipeline's making
  (`XPC_ERROR_CONNECTION_INTERRUPTED`), and the runtime returns the failure
  (`CommandBufferError::PipelineCreation`, or `Error::Vulkan` at an indirect command's commit).
  Metal runs the AIR, which gets no such rewrite (case `authored-four-iteration-loop`). No form of
  the loop is known that avoids the rewrite for every such loop.
- **An anisotropic sampler without a mip filter whose min and mag filters differ, through Metal's
  own sampler (MoltenVK).** Vulkan has no sampler without mipmapping, so the runtime hands MoltenVK
  a nearest mip filter with a level clamp: `[0, 0]` where the two filters agree, which is Metal's
  `NotMipmapped` bit for bit, and `[0, 0.25]` where they differ, which chooses Metal's filter, but
  at `maxAnisotropy` 2 to 16 no clamp reads Metal's `NotMipmapped` answer. Announced as
  `NotMipmappedAnisotropicSampler` for a sampler state and `NotMipmappedAnisotropicFilter` for a
  constexpr sampler.
- **Dynamic texel offsets past 512 arms in one function, on a device without the `Offset`
  operand.** Without `maintenance8` (a sample) or `shaderImageGatherExtended` (a gather) a dynamic
  texel offset is a switch over the values Metal reads, each arm the access with that value as
  `ConstOffset` (16 arms a dynamic lane), exactly. A function whose accesses would take more than
  512 arms together is refused (`TexelOffset` or `GatherExtended`): a driver's making grows faster
  than the arms (on ANV 512 arms make in 0.63 s, 1536 in 5.9 s). Applying the offset in the shader
  is exact only where the shader filters, so a filtered sample past the bound has no exact form yet.
- **A fetched attachment's blend through the pixel interlock (RADV, ANV, NVIDIA, MoltenVK).** No
  Vulkan blend applies to the storage form's write, so the function computes Metal's blend: an
  extended-range or 16-bit float attachment's, and an 8-bit unorm one's (`R8Unorm`, `RG8Unorm`,
  `RGBA8Unorm`, `BGRA8Unorm`) of a four-lane output under any factor, operation, write mask and
  blend color. Refused (`FramebufferFetch`, `NotImplemented`): a factor of the second source; under
  fast math, a lerp, common factor or contracted pair into `R8Unorm` or `RG8Unorm`, which Metal
  sums in forms not measured; and a blend into any other format, `RGB10A2Unorm`, `BGR10A2Unorm`,
  `RGBA8Snorm`, `RGBA16Unorm`, `RGBA16Snorm`, `RGBA32Float` and the 8-bit `_sRGB` ones among them.
  A source-over survey on the M3 fits the 8-bit model to the 10-bit unorm and 8-bit snorm formats
  (with their clamps), and shows the 16-bit norms and `RGBA32Float` blending in `float` even for a
  `half4` output; their factor sets are not measured.
- **Format-less storage reads on a device without `shaderStorageImageReadWithoutFormat` (ANV).**
  A fetch through the pixel interlock of a 16-bit packed color attachment (`B5G6R5Unorm`,
  `A1BGR5Unorm`, `ABGR4Unorm`, `BGR5A1Unorm`), which no SPIR-V format names, is refused
  (`FramebufferFetch`); so is a read of an argument buffer's writable texture, whose heap classes
  are format-less (`ArgumentBufferTextures`). A fetch through a 16-bit unsigned view that packs
  the channels in the shader, and heap storage classes typed per format, are not built.
- **The formats from 630 to 645 under macOS 27.0.1.** The macOS 27.0.1 level answers as an Apple M3
  Max, which describes none of them (`BGRA10Unorm`, its packed, sRGB, PQ and HLG forms,
  `R10UintX6`, `RG10UintX12` and the rest): a texture of one is stopped and the linear alignment
  query refused, as on the M3 Max. An Apple M4 under 27.0.1 describes and makes them; no level the
  runtime has does.
- **Tessellation's triangle order.** On a device with `tessellationShader` but MoltenVK on an
  Apple GPU, the device's tessellator orders a patch's triangles, picks their first corners and
  joins its rings itself, announced as `TessellationTriangleOrder`
  (`tessellation-triangle-order`): every patch differs on the RTX 3090, 55-95% on RADV and
  lavapipe.
- **lavapipe aborts on the owned fixture `mesh_order`'s mesh functions.** Compiling
  `mesh_order_mesh` or `mesh_order_solo` ends the process inside lavapipe's compiler ("LLVM ERROR:
  Cannot emit physreg copy instruction"); the validation case `mesh_order_mesh-payload-mesh-draw`
  runs on a GPU only. The construct behind the abort is not isolated yet.
