//! The gate lanes of the driver deviations translation relies on, on the device the cfg selects,
//! under the validation layer. A deviation is a driver's measured answer where Vulkan leaves the
//! answer to it, keyed on the driver's release (`device::driver_facts`): MoltenVK 1.4.1 on Apple's
//! GPUs has twenty-three. A profile takes each at a measured release, and past the last measured
//! one with a notice (`DeviceProfile::takes_driver_deviation`). Each test here runs what relies on
//! one deviation and compares the device's answer with Metal's, so a venue whose driver release
//! changes the deviation fails at the gate. The sampler's deviation has its own lanes (`depth_compares`,
//! `mip_levels`, `nearest_positions`, `border_swizzles`), and the color writes theirs
//! (`color_writes`); these are the others, the cube route's in both stages:
//!
//! - **Lockstep** (`SimdgroupLockstep`): the owned fixture `rtl_lockstep`'s kernels read and write
//!   threadgroup words that only a simdgroup's lockstep orders, inside loops each thread leaves
//!   after its own count, where no barrier is uniform in the simdgroup. A profile that takes the
//!   deviation compiles them with no barrier there and requires the device's lockstep
//!   (`VkProperty::SimdgroupLockstep`); every other profile takes emulated simdgroups. Both give
//!   Metal's answer, `lockstep_divergent_loop` and `lockstep_until`: every device is checked.
//! - **The float functions** (`MetalFloatFunctions`): the committed math-library probe cases, with
//!   and without fast math, run as their Metal oracle ran them. Where the profile takes the
//!   deviation, every word of a function the translation hands to the device's instruction is
//!   Metal's (`PRECISE_DEVICE_WORDS`, `FAST_DEVICE_WORDS`). Elsewhere the translation's routines
//!   answer within Apple's bound, and C3 compares them; the test prints how many words match.
//! - **The cube routes** (`GraphicsCubeGradientRoute`, `GraphicsCubeGradientTangents` and
//!   `CubeGradientRoute`): the owned fixture `rtl_cube_grad_fragment` samples a cube and a cube
//!   array with explicit gradients through an anisotropic constexpr sampler in a fragment function,
//!   over the inputs of the committed kernel cases `anisotropic-cube-gradients-along-face-rows` and
//!   `-columns`, which run too. Where the profile takes a route's deviation, each pixel or kernel
//!   lane is the Metal kernel's sample of its lane on the Apple M3 Max, the lane's Metal
//!   environment; where it takes `GraphicsCubeGradientTangents` (MoltenVK under macOS 27, whose
//!   cube-gradient swizzle loses a `y` face's `z` gradient), so are the lanes on the `x` and `z`
//!   faces, and the `y` faces' are a near-equivalent (`CubeGradientFootprint`) the test prints.
//!   MoltenVK's graphics route is measured on the M3 Max under macOS 26.7 and under macOS 27.0.1
//!   (MoltenVK's device ID is the macOS release and the GPU family, which the M4 under macOS 27
//!   shares); under any other, the fragment's pipeline is refused, naming the unmeasured deviation
//!   `GraphicsCubeGradientRoute`. Elsewhere the sample is a near-equivalent
//!   (`CubeGradientFootprint`) or refused; the test prints which.
//! - **The `half` fused multiply-add** (`MetalHalfFma`): the committed cases of a probe
//!   (`fmx_fma_precise`, `fmx_fma_fast`), whose half word is a `half` `fma` and a `half` `a * b +
//!   c` that Metal fuses, over 300 triples whose `float` sum lands on a `half` midpoint, 200 where
//!   one rounding and two differ, the special values and random ones. Where the profile takes the
//!   deviation, the kernels take the device's 16-bit `Fma` and require `VkProperty::MetalHalfFma`,
//!   and every half word is the Metal row's of the same chip. Elsewhere the `half` routine rounds
//!   the `float` sum and C3 compares it; the test prints how many half words match.
//! - **The `float` denormals** (`MetalFloatDenormals`): the NaN model's denormal probes, with and
//!   without fast math. Where the profile takes the deviation, the kernels leave `+ - *`, the
//!   comparisons, `ldexp`, `sqrt` and `rsqrt` to the device and require
//!   `VkProperty::MetalFloatDenormals`; elsewhere the module reads such operands and flushes such
//!   results itself. Both give Metal's answer, so every device is checked on those operations'
//!   words; `floor` and `ceil` read their operand themselves on every device. a probe cases check
//!   the fused pair's NaN and the lone product's flushed denormal on every device too.
//! - **The `half` quantize** (`MetalHalfQuantize`): the NaN model's narrowing, conversion and
//!   `half` cases. Where the profile takes the deviation, a `half` rounding is the device's
//!   `OpQuantizeToF16` alone and the kernels require `VkProperty::MetalHalfQuantize`; elsewhere the
//!   module rounds on the bits in integer arithmetic. Both give Metal's largest `half` there, so
//!   every device is checked on those words.
//! - **The `half` conversion** (`MetalHalfConversion`): the same cases, and the owned `half` edge
//!   fixtures on every word. Where the profile takes the deviation, a `half` rounding is the
//!   device's conversion to 16 bits and back, with no quantize, and the kernels require
//!   `VkProperty::MetalHalfConversion` and not `MetalHalfQuantize`; elsewhere they require neither
//!   or the quantize's fact as above.
//! - **The minimum and maximum family** (`MetalMinMax`): the NaN model's `fmin`, `fmax`, `min3`,
//!   `max3`, `clamp` and `median3` cases, at NaN operands of every kind and at zeros and denormals,
//!   with and without fast math. Where the profile takes the deviation, Metal's minimum, maximum
//!   and clamp are the device's `NMin`, `NMax` and `NClamp` and the kernels require
//!   `VkProperty::MetalMinMax`; elsewhere the module picks the operand itself. Both give Metal's
//!   answer, so every device is checked on every word.
//! - **The `half` arithmetic** (`MetalHalfArithmetic`): the NaN model's narrowing and `half`
//!   arithmetic cases, with and without fast math, and the `half` denormal arithmetic case. Where
//!   the profile takes the deviation, a `half` `+`, `-` or `*` is the device's 16-bit instruction,
//!   kept apart, and the kernels require `VkProperty::MetalHalfArithmetic`; elsewhere the module
//!   computes in `float` and rounds to `half` itself. Both give Metal's answer, so every device
//!   is checked on every word.
//! - **The fragment simdgroups** (`MetalFragmentSimdgroups`): the owned fixture
//!   `gsm2_fragment_lanes` writes a fragment's `[[thread_index_in_simdgroup]]`, read as a `uint`
//!   and a `ushort` parameter and as a program-scope `uint`, over three draws into a 32x32 target:
//!   one triangle over it whole, one over part of it, and a 5x5 grid of quads. Where the profile
//!   takes the deviation, the lane is the subgroup lane and every pixel holds Metal's lane of the
//!   same draw (the quads of an aligned 8x4 block in row-major order where one primitive covers it,
//!   the rasterizer's packing elsewhere, measured on the M3 Max and the M4 alike). Elsewhere, on a
//!   device whose fragment stage runs 32-lane subgroups, the lane is the device's own grouping's,
//!   which the lane checks only for coverage and range; on any other device each pipeline state is
//!   refused `SimdWidth`, needing `VkProperty::MetalFragmentSimdgroups` or a 32-lane stage. A
//!   fourth function writes the lane beside shuffles of it and the ballot's first lane, which must
//!   name the same lanes on every device that makes it. The fixture's `[[threads_per_simdgroup]]`
//!   parameter draws 32 on every device.
//! - **The vertex simdgroups** (`MetalVertexSimdgroups`): the committed cases of the owned fixtures
//!   `rvc_simd_vertex` (a sum, shuffles and a ballot over forty points) and `rvc_simd_vertex_fill`
//!   (a shuffle-and-fill down with an explicit width, which reads the lane), and the
//!   `gsm2_vertex_votes_shifts` (votes, shuffles up and down and a fill up, over full and partial
//!   simdgroups). Where the profile takes the deviation (MoltenVK, whose `subgroupSupportedStages`
//!   leaves the vertex stage out), each pipeline requires `VkProperty::MetalVertexSimdgroups`, the
//!   lane is the active-lane count below the invocation's, every word is the Metal row's, and the
//!   runtime announces `UnadvertisedVertexSimdgroups` once, after which the validation layer's
//!   report of the stage passes. Elsewhere the device's own grouping draws, which the test prints,
//!   or each state is refused, typed, where the vertex stage has no subgroups.

use super::degrade_notices::{counts as notice_counts, on_device};
use super::{
    buffer_with, public_case, read_buffer, run_case, validated_device, CaseResources, Messages,
    PublicCase, ScratchCache, METAL_ENVIRONMENT, WAIT,
};
use crate::air_library::AirLibrary;
use crate::case::{AuthoredCase, OutputSelection, Primitive};
use crate::gpu::TEST_DEVICE;
use crate::metal::ORACLE_ABI;
use crate::observation::MetalObservation;
use crate::store::CorpusStore;
use base64::Engine as _;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::{FloatControl, FloatWidth, VkFeature, VkProperty};
use metal2vulkan::error::{Capability, DriverDeviation, Error as CompileError, Feature, Needs};
use metal2vulkan::library::Library;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor,
    RenderPipelineDescriptor, Stage, StageSource,
};
use metal2vulkan::vk::DriverId;
use metal2vulkan::Budget;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBufferStatus, Degradation, Device,
    Error as RuntimeError, LoadAction, Origin, PipelineOptions, RenderPassDescriptor,
    ResourceOptions, Size, StoreAction, TextureDescriptor,
};
use std::path::{Path, PathBuf};

/// The owned fixture at `name` under `validation/fixtures/public/engine`.
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/public/engine")
        .join(name)
}

/// Whether `device`'s profile takes `deviation`.
fn takes(device: &Device, deviation: DriverDeviation) -> bool {
    device
        .compile_device()
        .profile()
        .takes_driver_deviation(deviation)
}

// ---------------------------------------------------------------------------------------------
// Lockstep.

/// The threads of `rtl_lockstep`'s threadgroup: two simdgroups.
const THREADS: usize = 64;

/// `rtl_lockstep_divergent_loop`'s iteration counts, 1 to 5.
fn counts() -> Vec<u32> {
    (0..THREADS as u32).map(|t| 1 + (t * 7 + 3) % 5).collect()
}

/// `rtl_lockstep_until`'s limits.
fn limits() -> Vec<u32> {
    (0..THREADS as u32).map(|t| 16 + (t * 37) % 97).collect()
}

/// Each simdgroup's 32 threads run each iteration together: the threads still in the loop all
/// read, then all write. `step` gives a thread's iteration from the words as the iteration found
/// them: `None` where the thread leaves the loop, else its new word.
fn in_lockstep(words: &mut [u32], mut step: impl FnMut(usize, u32, &[u32]) -> Option<u32>) {
    let mut active = vec![true; words.len()];
    let mut iteration = 0;
    while active.iter().any(|&a| a) {
        let before = words.to_vec();
        for t in 0..words.len() {
            if active[t] {
                match step(t, iteration, &before) {
                    Some(word) => words[t] = word,
                    None => active[t] = false,
                }
            }
        }
        iteration += 1;
    }
}

/// Metal's answer of `rtl_lockstep_divergent_loop`: thread t's word after `counts[t]` iterations
/// of `words[t] = words[t ^ 16] * 3 + i + 1`, the reads of an iteration before its writes.
/// Measured on the Apple M3 Max and the Apple M4.
fn lockstep_divergent_loop() -> Vec<u32> {
    let counts = counts();
    let mut words: Vec<u32> = (0..THREADS as u32).collect();
    in_lockstep(&mut words, |t, i, before| {
        (i < counts[t]).then(|| {
            before[t ^ 16]
                .wrapping_mul(3)
                .wrapping_add(i)
                .wrapping_add(1)
        })
    });
    let mut out = words;
    out.resize(2 * THREADS, 0xeeee_eeee);
    out
}

/// Metal's answer of `rtl_lockstep_until`: thread t adds the word of the thread 8 lanes on in its
/// simdgroup to its own while its own is below `limits[t]`, at most 8 times, every thread still in
/// the loop reading before any writes; then its word and its count. Measured as
/// `lockstep_divergent_loop` is.
fn lockstep_until() -> Vec<u32> {
    let limits = limits();
    let mut words: Vec<u32> = (1..=THREADS as u32).collect();
    let mut steps = vec![0u32; THREADS];
    in_lockstep(&mut words, |t, _, before| {
        if before[t] >= limits[t] || steps[t] >= 8 {
            return None;
        }
        steps[t] += 1;
        let partner = ((t + 8) & 31) | (t & 32);
        Some(before[t].wrapping_add(before[partner]))
    });
    words.extend(steps);
    words
}

/// The words `kernel` of `rtl_lockstep` writes for `input` on `device`, one 64-thread
/// threadgroup.
fn lockstep_run(
    device: &Device,
    library: &metal2vulkan_runtime::Library,
    kernel: &str,
    input: &[u32],
) -> Vec<u32> {
    let function = library.new_function_with_name(kernel).unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let pipeline = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let bytes: Vec<u8> = input.iter().flat_map(|x| x.to_le_bytes()).collect();
    let a = buffer_with(device, &bytes).unwrap();
    let o = buffer_with(device, &[0xee; 8 * THREADS]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(Some(&a), 0, 0).unwrap();
    encoder.set_buffer(Some(&o), 0, 1).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(THREADS as u64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&o, 0, 8 * THREADS as u64)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Whether `kernel` of the fixture `bitcode`, as `device`'s compile layer compiles it, relies on
/// the device's lockstep (`VkProperty::SimdgroupLockstep`).
fn requires_lockstep(device: &Device, bitcode: &[u8], kernel: &str) -> bool {
    let library = Library::from_air(bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function(kernel).unwrap());
    let inputs = VariantKey::new().lowering_inputs(Stage::Kernel);
    device
        .compile_device()
        .compile_function(&function, &inputs, &Budget::standard())
        .unwrap()
        .requirements()
        .properties
        .contains(&VkProperty::SimdgroupLockstep)
}

/// Both kernels answer as Metal's lockstep orders them on every device: through the device's own
/// lockstep where the profile takes the deviation (the compiled kernel then requires it), and
/// through emulated simdgroups elsewhere.
#[test]
fn threadgroup_words_only_lockstep_orders_answer_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lockstep = takes(&device, DriverDeviation::SimdgroupLockstep);
        let bitcode = llvm_assemble(&fixture("rtl_lockstep.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        for (kernel, input, metal) in [
            (
                "rtl_lockstep_divergent_loop",
                counts(),
                lockstep_divergent_loop(),
            ),
            ("rtl_lockstep_until", limits(), lockstep_until()),
        ] {
            assert_eq!(
                requires_lockstep(&device, &bitcode, kernel),
                lockstep,
                "{kernel} on {name}: the kernel requires the device's lockstep exactly where the \
                 profile takes it"
            );
            let read = lockstep_run(&device, &library, kernel, &input);
            assert_eq!(
                read, metal,
                "{kernel} on {name} (the device's lockstep: {lockstep})"
            );
            eprintln!(
                "lockstep {kernel} on {name}: Metal's answer (the device's lockstep: {lockstep})"
            );
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The float functions.

/// `math-library-accuracy-at-1024-operand-pairs-without-fast-math` (`trn_math_precise`): each
/// thread writes 32 words.
const PRECISE: &str = "math-library-accuracy-at-1024-operand-pairs-without-fast-math";
/// `math-library-accuracy-at-1024-operand-pairs-under-fast-math` (`trn_math_fast`).
const FAST: &str = "math-library-accuracy-at-1024-operand-pairs-under-fast-math";

/// The words of a thread of `PRECISE` whose function the device's instruction answers where
/// the profile takes the deviation: `sin`, `cos`, `tan`, `sincos`'s two, `asin`, `acos`, `atan`,
/// `atan2(y, x)`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `exp2`, `log`, `log2`,
/// `powr`, `sqrt`, `rsqrt`, `fmod`, `x / y` and `atan2(x, y)`. Not the compositions the translation
/// builds on every device: `exp10`, `log10` and `pow` (words 17, 20, 21) and `sinpi`, `cospi` and
/// `tanpi` (26 to 28); nor word 31, the operand `y`.
const PRECISE_DEVICE_WORDS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 29, 30,
];

/// The same for `FAST`, in a fast-math entry: the fast `sin`, `cos`, `tan`, `sincos`'s two,
/// `asin`, `acos` and `atan`, the fast hyperbolic functions' own instructions (`sinh`, `cosh`,
/// `asinh`, `acosh`, `atanh`), `tanh` (SPIRV-Cross writes `precise::tanh`), `exp2` and `log2`
/// (Metal's fast ones are its precise ones), `sqrt`, `rsqrt` and `x / y` (the device's division
/// in a fast-math entry). Not `atan2` (words 8 and 30: SPIRV-Cross writes `precise::atan2`, so the
/// translation composes Metal's fast one), the fast `exp`, `log`, `exp10`, `log10`, `pow` and
/// `powr` (15, 17, 18, 20 to 22: compositions of the device's `exp2` and `log2`), `fmod` (25: a
/// composition of the device's reciprocal), nor `sinpi`, `cospi`, `tanpi` and `y`.
const FAST_DEVICE_WORDS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 19, 23, 24, 29,
];

/// The words of `words` in each 32-word thread of `output` and `metal` that differ: `(thread,
/// word, device, Metal)`.
fn differing(output: &[u8], metal: &[u8], words: &[usize]) -> Vec<(usize, usize, u32, u32)> {
    differing_of(output, metal, 32, words)
}

/// The words of `words` in each `per_thread`-word thread of `output` and `metal` that differ:
/// `(thread, word, device, Metal)`.
fn differing_of(
    output: &[u8],
    metal: &[u8],
    per_thread: usize,
    words: &[usize],
) -> Vec<(usize, usize, u32, u32)> {
    let word =
        |bytes: &[u8], i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!(output.len(), metal.len());
    let threads = metal.len() / (4 * per_thread);
    let mut differ = Vec::new();
    for thread in 0..threads {
        for &w in words {
            let at = per_thread * thread + w;
            let (a, b) = (word(output, at), word(metal, at));
            if a != b {
                differ.push((thread, w, a, b));
            }
        }
    }
    differ
}

/// Where the profile takes the deviation, every word the device's instruction answers is
/// Metal's, in both math modes (Metal's rows of both cases are the same on the Apple M3 Max and
/// the Apple M4).
#[test]
fn float_functions_the_device_answers_are_metals() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let metals = takes(&device, DriverDeviation::MetalFloatFunctions);
        for (case_name, words) in [(PRECISE, PRECISE_DEVICE_WORDS), (FAST, FAST_DEVICE_WORDS)] {
            let case = public_case(case_name).unwrap();
            let run = run_case(&device, &case).unwrap();
            let differ = differing(&run.output, &case.metal_output, words);
            let all = differing(
                &run.output,
                &case.metal_output,
                &(0..32).collect::<Vec<_>>(),
            );
            eprintln!(
                "float functions {case_name} on {name} (Metal's functions: {metals}): {} of {} \
                 device words and {} of {} words differ from Metal",
                differ.len(),
                words.len() * 1024,
                all.len(),
                32 * 1024
            );
            if metals {
                assert!(
                    differ.is_empty(),
                    "{case_name} on {name}: {} device words differ from Metal (thread, word, \
                     device, Metal): {:x?}",
                    differ.len(),
                    &differ[..differ.len().min(24)]
                );
            }
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The graphics cube route.

/// The committed kernel cases whose inputs the fragment function samples.
const CUBE_CASES: [&str; 2] = [
    "anisotropic-cube-gradients-along-face-rows",
    "anisotropic-cube-gradients-along-face-columns",
];

/// The lanes of each case: the kernel's grid.
const LANES: u64 = 72;

/// The output of `case`'s Metal row in `environment`.
fn metal_row(case: &PublicCase, environment: &str) -> Vec<u8> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let row = MetalObservation::current(
        store.read_metal_of_case(&case.case).unwrap(),
        &case.case,
        environment,
        ORACLE_ABI,
    )
    .unwrap();
    base64::engine::general_purpose::STANDARD
        .decode(row.observed_output().unwrap())
        .unwrap()
}

/// The fragment function's 72x2 `RGBA32Float` texels over `case`'s inputs on `device`, each
/// pixel `(i, layer)` reordered to the kernel's word `2i + layer`; or why pipeline creation
/// refused.
fn fragment_samples(device: &Device, case: &PublicCase) -> Result<Vec<u8>, RuntimeError> {
    let bitcode = llvm_assemble(&fixture("rtl_cube_grad_fragment.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = |name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("rtl_cube_cover")));
    descriptor.fragment = Some(StageSource::Function(function("rtl_cube_grad_fragment")));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    let pipeline = device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())?;
    let resources = CaseResources::new(device, case).unwrap();
    let input = buffer_with(device, &case.compute.buffers[&0]).unwrap();
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, LANES, 2, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    attachment.store_action = StoreAction::Store;
    let length = 16 * LANES * 2;
    let readback = buffer_with(device, &vec![0; length as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    encoder
        .set_fragment_texture(Some(&resources.textures[&0]), 0)
        .unwrap();
    encoder
        .set_fragment_texture(Some(&resources.textures[&1]), 1)
        .unwrap();
    encoder.set_fragment_buffer(Some(&input), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(LANES, 2, 1),
        &readback,
        0,
        16 * LANES,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let texels = read_buffer(&readback, 0, length).unwrap();
    let texel = |layer: u64, i: u64| {
        let at = (16 * (LANES * layer + i)) as usize;
        texels[at..at + 16].to_vec()
    };
    Ok((0..LANES)
        .flat_map(|i| [texel(0, i), texel(1, i)])
        .flatten()
        .collect())
}

/// The blocks of 12 lanes on a `y` face: each case's lanes take the faces in blocks of 12, in the
/// order `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`.
const Y_FACE_BLOCKS: [usize; 2] = [2, 3];

/// The samples of `samples` that differ from `metal` (16 bytes each), and how many per block of
/// 12 lanes (two samples a lane: each case's lanes take the faces in blocks of 12).
fn differing_samples(samples: &[u8], metal: &[u8]) -> (Vec<usize>, Vec<usize>) {
    let differ: Vec<usize> = (0..metal.len() / 16)
        .filter(|&t| samples.get(16 * t..16 * t + 16) != metal.get(16 * t..16 * t + 16))
        .collect();
    let mut per_block = vec![0; metal.len() / 16 / 24];
    for &t in &differ {
        if let Some(n) = per_block.get_mut(t / 24) {
            *n += 1;
        }
    }
    (differ, per_block)
}

/// Where the profile takes each route's deviation, the fragment function's samples (the graphics
/// route, `GraphicsCubeGradientRoute`) and the committed kernel's (the compute route,
/// `CubeGradientRoute`) are the Metal kernel's of the same lanes on the Apple M3 Max. Where it
/// takes `GraphicsCubeGradientTangents` instead, the fragment's samples on the `x` and `z` faces
/// are; its `y` faces' are printed. On MoltenVK where a route is unmeasured (any Apple GPU and
/// macOS release but the M3 Max's under macOS 26.7 and 27: MoltenVK's device ID is both), the
/// function emulates Metal's footprint, and its pipeline is made: its samples are printed. Every
/// case and route is measured and printed before any is judged.
#[test]
fn cube_gradients_through_an_anisotropic_sampler_answer_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let mut failures = Vec::new();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let graphics = takes(&device, DriverDeviation::GraphicsCubeGradientRoute);
        let tangents = takes(&device, DriverDeviation::GraphicsCubeGradientTangents);
        let compute = takes(&device, DriverDeviation::CubeGradientRoute);
        let moltenvk = device.properties().description.driver_id == Some(DriverId::MOLTENVK);
        let environment = METAL_ENVIRONMENT;
        for case_name in CUBE_CASES {
            let case = public_case(case_name).unwrap();
            let metal = metal_row(&case, environment);
            let fragment = fragment_samples(&device, &case).map_err(|error| format!("{error}"));
            let kernel = run_case(&device, &case).map(|run| run.output);
            for (stage, taken, y_faces_near, answer) in [
                ("fragment", graphics || tangents, tangents, fragment),
                ("kernel", compute, false, kernel),
            ] {
                let verdict = match answer {
                    Ok(samples) => {
                        let (differ, per_block) = differing_samples(&samples, &metal);
                        eprintln!(
                            "cube route {stage} {case_name} on {name} (the measured route: \
                             {taken}, the y faces near: {y_faces_near}): {} of {} samples differ \
                             from {environment}'s kernel, per 12 lanes {per_block:?}: {differ:?}",
                            differ.len(),
                            metal.len() / 16
                        );
                        let judged: Vec<usize> = differ
                            .into_iter()
                            .filter(|t| !(y_faces_near && Y_FACE_BLOCKS.contains(&(t / 24))))
                            .collect();
                        match taken && !judged.is_empty() {
                            true => Some(format!("samples {judged:?} differ")),
                            false => None,
                        }
                    }
                    Err(why) => {
                        eprintln!(
                            "cube route {stage} {case_name} on {name} (the measured route: \
                             {taken}): {why}"
                        );
                        (taken || moltenvk).then_some(why)
                    }
                };
                if let Some(why) = verdict {
                    failures.push(format!("{stage} {case_name} on {name}: {why}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The half fused multiply-add.

/// The committed cases of a probe: 1024 threads of four words each, word 2 the `half` `fma(a, b,
/// c)` in its low half and the `half` `a * b + c` (`llvm.fmuladd` without fast math, a contracted
/// pair under it, both fused by Metal) in its high half.
const HALF_FMA_CASES: [&str; 2] = [
    "fused-multiply-add-rounding-and-fmuladd-without-fast-math",
    "fused-multiply-add-rounding-and-contraction-under-fast-math",
];

/// The word of a [`HALF_FMA_CASES`] thread that holds its two `half` results.
const HALF_WORD: usize = 2;

/// Whether `case`'s kernel, as `device`'s compile layer compiles it, requires `property`: for
/// [`VkProperty::MetalHalfFma`], whether it takes the device's 16-bit `Fma`.
fn requires_property(device: &Device, case: &PublicCase, property: VkProperty) -> bool {
    let library = Library::from_air(&case.bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function(&case.case.entry).unwrap());
    let inputs = VariantKey::new().lowering_inputs(Stage::Kernel);
    device
        .compile_device()
        .compile_function(&function, &inputs, &Budget::standard())
        .unwrap()
        .requirements()
        .properties
        .contains(&property)
}

/// Whether `case`'s kernel rounds to `half` under the float controls that make the device's
/// conversion Metal's: it declares `DenormPreserve` for 16 bits, which no other form declares, and
/// so requires that property.
fn rounds_under_float_controls(device: &Device, case: &PublicCase) -> bool {
    requires_property(
        device,
        case,
        VkProperty::FloatControl(FloatControl::DenormPreserve, FloatWidth::F16),
    )
}

/// Where the profile takes the deviation, both cases' kernels take the device's 16-bit `Fma`
/// (they require `VkProperty::MetalHalfFma`) and every thread's two `half` results are the Metal
/// row's of the same chip, with and without fast math. Elsewhere neither kernel requires it.
#[test]
fn half_fused_multiply_adds_the_device_answers_are_metals() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let metals = takes(&device, DriverDeviation::MetalHalfFma);
        for case_name in HALF_FMA_CASES {
            let case = public_case(case_name).unwrap();
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalHalfFma),
                metals,
                "{case_name} on {name}: the kernel takes the device's 16-bit Fma exactly where \
                 the profile takes the deviation"
            );
            let run = run_case(&device, &case).unwrap();
            let metal = metal_row(&case, METAL_ENVIRONMENT);
            let differ = differing_of(&run.output, &metal, 4, &[HALF_WORD]);
            eprintln!(
                "half fma {case_name} on {name} (the device's Fma: {metals}): {} of {} half \
                 words differ from Metal",
                differ.len(),
                metal.len() / 16
            );
            if metals {
                assert!(
                    differ.is_empty(),
                    "{case_name} on {name}: {} half words differ from Metal (thread, word, \
                     device, Metal): {:x?}",
                    differ.len(),
                    &differ[..differ.len().min(24)]
                );
            }
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The `float` denormals, the `half` quantize, and the fused pair's NaN.

/// The NaN model's denormal probe (`nan_denormals`, `nan_denormals_fast`): 256 threads of 48
/// words over operand triples rich in denormals of both signs and the smallest normals; the
/// second is the first's program under fast math.
const DENORMAL_PROBES: [(&str, bool); 2] = [
    ("nan-model-float-denormals-per-operation", false),
    (
        "nan-model-float-denormals-per-operation-under-fast-math",
        true,
    ),
];

/// The words of a [`DENORMAL_PROBES`] thread whose operations read `float` denormals as Metal
/// does on every profile: `a + b`, `a - b`, `a * b`, `0 - a`, `a < b`, `a == b`, `floor`, `ceil`,
/// `trunc`, `round`, `rint`, `ldexp`, `step`, `(a * b) * c`, `a < b ? a : b`, `-(a * b)` and
/// `fabs(a - b)`; the quotients ([`DENORMAL_PROBE_QUOTIENTS`]); `sqrt` and `rsqrt`
/// ([`DENORMAL_PROBE_ROOTS`]) where Metal answers a zero, an infinity or a NaN; and `pow` and
/// `powr` ([`DENORMAL_PROBE_POWERS`]) where it answers a zero, a denormal, an infinity or a NaN.
/// Not the fused multiply-adds (4, 5), `dot` and `length` (36, 37) and `mix` (38): their fused
/// sums flush at the tininess edge before rounding on Metal, which no device measured matches on
/// every lane, nor the math library's functions.
const DENORMAL_PROBE_WORDS: &[usize] = &[
    0, 1, 2, 7, 10, 11, 12, 13, 14, 15, 16, 35, 39, 43, 45, 46, 47,
];

/// The quotient words of a [`DENORMAL_PROBES`] thread: `a / b`, `smoothstep` and `(a / b) * c`.
/// Under fast math each divides by Metal's fast reciprocal, which a profile whose `float`
/// functions are not Metal's and that has no `OpFmaKHR` (`shaderFmaFloat32`) takes as the
/// device's `OpFDiv(1, y)`, within Vulkan's 2.5 ulp, not `RN(1 / y)` (the translator's
/// `divide::Reciprocal::Device`, recorded `PrecisionNote::DeviceArithmetic`): on such a profile
/// the fast probe's quotients are checked where Metal answers a zero, a denormal, an infinity or
/// a NaN, and everywhere else every word of them.
const DENORMAL_PROBE_QUOTIENTS: &[usize] = &[3, 40, 44];

/// The `pow` and `powr` words of a [`DENORMAL_PROBES`] thread: checked where Metal answers a
/// zero, a denormal, an infinity or a NaN, among them `pow(x, 1)` of a denormal `x`, which
/// precise `pow` answers as `x` and `fast::pow` as the zero of its sign. Their other words are the
/// math library's values at normal operands.
const DENORMAL_PROBE_POWERS: &[usize] = &[32, 33];

/// The `sqrt` and `rsqrt` words of a [`DENORMAL_PROBES`] thread: checked where Metal answers a
/// zero, an infinity or a NaN (a zero, denormal or negative operand). Their other words are the
/// device's roots' rounding, not their denormals: NVIDIA's fast `sqrt` and `rsqrt` of 2 and 0.5
/// are an ulp below Metal's (the committed rows of 615.71.09, 16 lanes each).
const DENORMAL_PROBE_ROOTS: &[usize] = &[20, 21];

/// Both probes answer Metal's `float` denormals on every profile: the device's where the profile
/// takes the deviation (the kernels then require `VkProperty::MetalFloatDenormals`), the
/// module's own flushes elsewhere. Every word of [`DENORMAL_PROBE_WORDS`], each word of
/// [`DENORMAL_PROBE_QUOTIENTS`] (in the fast probe on a profile whose fast reciprocal is the
/// device's, where Metal answers a zero, a denormal, an infinity or a NaN), each word of
/// [`DENORMAL_PROBE_ROOTS`] where Metal answers a zero, an infinity or a NaN, and each word of
/// [`DENORMAL_PROBE_POWERS`] where it answers one of those or a denormal, is the Metal row's of
/// the same chip; the test prints how many of the other words differ.
#[test]
fn float_denormals_read_as_metal_reads_them() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let metals = takes(&device, DriverDeviation::MetalFloatDenormals);
        // Whether Metal's fast reciprocal is the device's `OpFDiv(1, y)`: neither Metal's own
        // float functions nor `OpFmaKHR`, whose refinement makes it `RN(1 / y)`.
        let device_reciprocal = !takes(&device, DriverDeviation::MetalFloatFunctions)
            && !device
                .compile_device()
                .profile()
                .enabled()
                .features
                .contains(&VkFeature::ShaderFmaFloat32);
        let mut failures = Vec::new();
        for (case_name, fast) in DENORMAL_PROBES {
            let case = public_case(case_name).unwrap();
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalFloatDenormals),
                metals,
                "{case_name} on {name}: the kernel leaves its denormals to the device exactly \
                 where the profile takes the deviation"
            );
            let run = run_case(&device, &case).unwrap();
            let metal = metal_row(&case, METAL_ENVIRONMENT);
            assert_eq!(run.output.len(), metal.len(), "{case_name} on {name}");
            let word = |bytes: &[u8], i: usize| {
                u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
            };
            let special = |w: u32| w & 0x7FFF_FFFF == 0 || w & 0x7F80_0000 == 0x7F80_0000;
            let edge = |w: u32| special(w) || w & 0x7F80_0000 == 0;
            let (mut checked, mut differ) = (0, Vec::new());
            let mut by_word = std::collections::BTreeMap::<usize, usize>::new();
            for i in 0..metal.len() / 4 {
                let w = i % 48;
                let (got, want) = (word(&run.output, i), word(&metal, i));
                let quotient = DENORMAL_PROBE_QUOTIENTS.contains(&w);
                if DENORMAL_PROBE_WORDS.contains(&w)
                    || (quotient && !(fast && device_reciprocal))
                    || (quotient && edge(want))
                    || (DENORMAL_PROBE_ROOTS.contains(&w) && special(want))
                    || (DENORMAL_PROBE_POWERS.contains(&w) && edge(want))
                {
                    checked += 1;
                    if got != want {
                        differ.push((i / 48, w, got, want));
                    }
                } else if got != want {
                    *by_word.entry(w).or_default() += 1;
                }
            }
            eprintln!(
                "float denormals {case_name} on {name} (the device's: {metals}; the device's \
                 fast reciprocal: {device_reciprocal}): {} of {checked} checked words differ \
                 from Metal; other words differing, by word: {by_word:?}",
                differ.len()
            );
            if !differ.is_empty() {
                failures.push(format!(
                    "{case_name} on {name}: {} words differ from Metal (thread, word, device, \
                     Metal): {:x?}",
                    differ.len(),
                    &differ[..differ.len().min(24)]
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// a probe cases ([`HALF_FMA_CASES`]): every word, `fma(a, b, c)`, `a * b + c`, the two `half`
/// results and the lone product `a * w`, is the Metal row's of the same chip on every profile: the
/// fused pair answers Metal's canonical NaN and the product Metal's flushed denormal (2.2).
#[test]
fn fused_pairs_answer_metals_nans_and_denormals() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let mut failures = Vec::new();
        for case_name in HALF_FMA_CASES {
            let case = public_case(case_name).unwrap();
            let run = run_case(&device, &case).unwrap();
            let metal = metal_row(&case, METAL_ENVIRONMENT);
            let differ = differing_of(&run.output, &metal, 4, &[0, 1, 2, 3]);
            eprintln!(
                "fused pairs {case_name} on {name}: {} of {} words differ from Metal",
                differ.len(),
                metal.len() / 4
            );
            if !differ.is_empty() {
                failures.push(format!(
                    "{case_name} on {name}: {} words differ from Metal (thread, word, device, \
                     Metal): {:x?}",
                    differ.len(),
                    &differ[..differ.len().min(24)]
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// The NaN model's cases whose words round a `float` near 65520 to `half`: the narrowing, the
/// vector conversions and the `half` operations, each without and under fast math.
const HALF_EDGE_CASES: [&str; 6] = [
    "nan-model-float-to-half-narrowing-and-half-arithmetic",
    "nan-model-float-to-half-narrowing-and-half-arithmetic-under-fast-math",
    "nan-model-vector-float-half-conversions-of-nan-operands",
    "nan-model-vector-float-half-conversions-of-nan-operands-under-fast-math",
    "nan-model-half-operations-and-widening-of-nan-operands",
    "nan-model-half-operations-and-widening-of-nan-operands-under-fast-math",
];

/// Every word of [`HALF_EDGE_CASES`] whose Metal answer holds the largest `half` (`0x7BFF` or
/// `0xFBFF` in either half, or the `float` 65504 of either sign) is the Metal row's of the same
/// chip on every profile: the device's conversion to 16 bits and back where the profile takes
/// `DriverDeviation::MetalHalfConversion` (the kernels then require
/// `VkProperty::MetalHalfConversion` and hold no quantize), else that conversion under the float
/// controls that make it Metal's where the profile has them (the kernels then declare them and
/// require neither fact), else the device's quantize where the profile takes `MetalHalfQuantize`
/// (the kernels then require `VkProperty::MetalHalfQuantize`), integer arithmetic elsewhere.
#[test]
fn half_quantize_rounds_as_metal_at_the_edge() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        // A profile that rounds through the device's conversion holds no quantize to rely on.
        let converts = takes(&device, DriverDeviation::MetalHalfConversion);
        let metals = takes(&device, DriverDeviation::MetalHalfQuantize) && !converts;
        let word = |bytes: &[u8], i: usize| {
            u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
        };
        let largest = |w: u32| {
            [w & 0xFFFF, w >> 16]
                .iter()
                .any(|h| matches!(h, 0x7BFF | 0xFBFF))
                || matches!(w, 0x477F_E000 | 0xC77F_E000)
        };
        let mut failures = Vec::new();
        for case_name in HALF_EDGE_CASES {
            let case = public_case(case_name).unwrap();
            let controlled = rounds_under_float_controls(&device, &case);
            assert!(
                !(converts && controlled),
                "{case_name} on {name}: the measured conversion takes no float controls"
            );
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalHalfQuantize),
                metals && !controlled,
                "{case_name} on {name}: the kernel takes the device's quantize alone exactly \
                 where the profile takes the deviation and has not the float controls"
            );
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalHalfConversion),
                converts,
                "{case_name} on {name}: the kernel rounds through the device's conversion \
                 exactly where the profile takes the deviation"
            );
            let run = run_case(&device, &case).unwrap();
            let metal = metal_row(&case, METAL_ENVIRONMENT);
            assert_eq!(run.output.len(), metal.len(), "{case_name} on {name}");
            let edge: Vec<usize> = (0..metal.len() / 4)
                .filter(|&i| largest(word(&metal, i)))
                .collect();
            let differ: Vec<(usize, u32, u32)> = edge
                .iter()
                .map(|&i| (i, word(&run.output, i), word(&metal, i)))
                .filter(|(_, a, b)| a != b)
                .collect();
            eprintln!(
                "half quantize {case_name} on {name} (the device's: {metals}, the device's \
                 conversion: {converts}, under float controls: {controlled}): {} of {} edge \
                 words differ from Metal",
                differ.len(),
                edge.len()
            );
            assert!(!edge.is_empty(), "{case_name}: no edge word");
            if !differ.is_empty() {
                failures.push(format!(
                    "{case_name} on {name}: edge words differ from Metal (word, device, Metal): \
                     {differ:x?}"
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// The owned denormal fixtures (`dfl_float_denormals`, `dfl_float_denormals_fast`): 256 threads of
/// 16 four-lane rows over every pair of a 32-value pool rich in denormals.
const DFL_DENORMAL_CASES: [&str; 2] = [
    "float-denormals-in-vector-arithmetic-comparisons-and-functions",
    "float-denormals-in-vector-arithmetic-comparisons-and-functions-under-fast-math",
];

/// The rows of a [`DFL_DENORMAL_CASES`] thread checked on every lane: `a + b`, `a - b`,
/// `a * b`, `fma`, `a * b + c`, the six comparisons, `floor`, `ceil`, `ldexp`, `(a * b) * c`,
/// `float(n) * a - c`, the three comparisons of results, `fmin(a, b) * c` and `step`. The fused
/// rows' lanes are clear of the tininess edge before rounding: Metal's rows are the operands
/// flushed, one rounding, and the result flushed, on every lane (and under fast math `a * b + c` is
/// two roundings, the product having other uses). `sqrt` (8) and `rsqrt` (9) are checked where
/// Metal answers a zero, an infinity or a NaN, at a zero, denormal or negative operand
/// ([`DFL_ROOT_ROWS`]).
const DFL_DENORMAL_ROWS: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15];

/// The `sqrt` and `rsqrt` rows of a [`DFL_DENORMAL_CASES`] thread: their other words are the
/// device's roots' rounding, not their denormals.
const DFL_ROOT_ROWS: &[usize] = &[8, 9];

/// The owned `half` edge fixtures (`dfl_half_edge`, `dfl_half_edge_fast`): 64 threads of 16 words,
/// every word a `half` pair rounded at the edge of the `half` range or among its denormals.
const DFL_HALF_EDGE_CASES: [&str; 2] = [
    "float-to-half-rounding-at-the-half-range-edge-and-denormals",
    "float-to-half-rounding-at-the-half-range-edge-and-denormals-under-fast-math",
];

/// The owned fixture of the precise trigonometric functions under fast-math flags
/// (`dfl_precise_trig_fast`): 64 threads of 12 words, NaN operands of each sign and payload among
/// them.
const DFL_TRIG_CASE: &str = "precise-trigonometric-functions-at-nan-operands-under-fast-math";

/// The owned fixtures answer Metal on every profile: in the denormal cases every lane of
/// [`DFL_DENORMAL_ROWS`] and the special lanes of [`DFL_ROOT_ROWS`], the kernels requiring
/// `VkProperty::MetalFloatDenormals` exactly where the profile takes the deviation; in the `half`
/// edge cases every word, the kernels requiring `VkProperty::MetalHalfConversion` exactly where the
/// profile takes that one, else neither where they round under the float controls, else
/// `VkProperty::MetalHalfQuantize` exactly where the profile takes that one;
/// and in [`DFL_TRIG_CASE`] every word whose Metal answer is a NaN (the functions' other words are
/// the math library's precision, not their NaNs). The test prints how many words of the other rows
/// differ, by row.
#[test]
fn owned_float_fixtures_answer_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let word = |bytes: &[u8], i: usize| {
            u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
        };
        let special = |w: u32| w & 0x7FFF_FFFF == 0 || w & 0x7F80_0000 == 0x7F80_0000;
        let mut failures = Vec::new();
        let denormals = takes(&device, DriverDeviation::MetalFloatDenormals);
        for case_name in DFL_DENORMAL_CASES {
            let case = public_case(case_name).unwrap();
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalFloatDenormals),
                denormals,
                "{case_name} on {name}: the kernel leaves its denormals to the device exactly \
                 where the profile takes the deviation"
            );
            let run = run_case(&device, &case).unwrap();
            let metal = metal_row(&case, METAL_ENVIRONMENT);
            assert_eq!(run.output.len(), metal.len(), "{case_name} on {name}");
            let (mut checked, mut differ) = (0, Vec::new());
            let mut others = std::collections::BTreeMap::<usize, usize>::new();
            for i in 0..metal.len() / 4 {
                let row = i % 64 / 4;
                let (got, want) = (word(&run.output, i), word(&metal, i));
                let check = DFL_DENORMAL_ROWS.contains(&row)
                    || (DFL_ROOT_ROWS.contains(&row) && special(want));
                if check {
                    checked += 1;
                    if got != want {
                        differ.push((i / 64, i % 64, got, want));
                    }
                } else if got != want {
                    *others.entry(row).or_default() += 1;
                }
            }
            eprintln!(
                "owned float denormals {case_name} on {name} (the device's: {denormals}): {} of \
                 {checked} checked words differ from Metal; other words differing, by row: \
                 {others:?}",
                differ.len()
            );
            if !differ.is_empty() {
                failures.push(format!(
                    "{case_name} on {name}: {} words differ from Metal (thread, word, device, \
                     Metal): {:x?}",
                    differ.len(),
                    &differ[..differ.len().min(24)]
                ));
            }
        }
        let converts = takes(&device, DriverDeviation::MetalHalfConversion);
        let quantize = takes(&device, DriverDeviation::MetalHalfQuantize) && !converts;
        for case_name in DFL_HALF_EDGE_CASES {
            let case = public_case(case_name).unwrap();
            let controlled = rounds_under_float_controls(&device, &case);
            assert!(
                !(converts && controlled),
                "{case_name} on {name}: the measured conversion takes no float controls"
            );
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalHalfQuantize),
                quantize && !controlled,
                "{case_name} on {name}: the kernel takes the device's quantize alone exactly \
                 where the profile takes the deviation and has not the float controls"
            );
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalHalfConversion),
                converts,
                "{case_name} on {name}: the kernel rounds through the device's conversion \
                 exactly where the profile takes the deviation"
            );
            let run = run_case(&device, &case).unwrap();
            let metal = metal_row(&case, METAL_ENVIRONMENT);
            let differ = differing_of(&run.output, &metal, 16, &(0..16).collect::<Vec<_>>());
            eprintln!(
                "owned half edge {case_name} on {name} (the device's quantize: {quantize}, \
                 conversion: {converts}, under float controls: {controlled}): {} of {} words \
                 differ from Metal",
                differ.len(),
                metal.len() / 4
            );
            if !differ.is_empty() {
                failures.push(format!(
                    "{case_name} on {name}: {} words differ from Metal (thread, word, device, \
                     Metal): {:x?}",
                    differ.len(),
                    &differ[..differ.len().min(24)]
                ));
            }
        }
        let case = public_case(DFL_TRIG_CASE).unwrap();
        let run = run_case(&device, &case).unwrap();
        let metal = metal_row(&case, METAL_ENVIRONMENT);
        assert_eq!(run.output.len(), metal.len(), "{DFL_TRIG_CASE} on {name}");
        let nan = |w: u32| w & 0x7F80_0000 == 0x7F80_0000 && w & 0x007F_FFFF != 0;
        let nans: Vec<usize> = (0..metal.len() / 4)
            .filter(|&i| nan(word(&metal, i)))
            .collect();
        let differ: Vec<(usize, usize, u32, u32)> = nans
            .iter()
            .map(|&i| (i / 12, i % 12, word(&run.output, i), word(&metal, i)))
            .filter(|(_, _, a, b)| a != b)
            .collect();
        eprintln!(
            "owned precise trig {DFL_TRIG_CASE} on {name}: {} of {} NaN words differ from Metal",
            differ.len(),
            nans.len()
        );
        assert!(!nans.is_empty(), "{DFL_TRIG_CASE}: no NaN word");
        if !differ.is_empty() {
            failures.push(format!(
                "{DFL_TRIG_CASE} on {name}: {} NaN words differ from Metal (thread, word, device, \
                 Metal): {:x?}",
                differ.len(),
                &differ[..differ.len().min(24)]
            ));
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Metal's `dot`: the fused chain of its products.

/// The owned `dot` fixtures (`nan_dot`, `nan_dot_fast`): 512 threads of 8 words, `dot` on two,
/// three and four `float` and `half` lanes and the shared-operand forms `dot(float2(a, b),
/// float2(b, c))` and `dot(float2(a, b), float2(b, a))`, without fast math and under the default
/// fast math.
const DOT_CASES: [&str; 2] = [
    "dot-of-two-to-four-float-and-half-lanes",
    "dot-of-two-to-four-float-and-half-lanes-under-fast-math",
];

/// Both `dot` cases answer Metal on every profile, every word: Metal's fused chain by the multiply
/// and the profile's fused multiply-adds, or on `float` the device's `OpDot` where it reaches
/// Metal's own `dot` (MoltenVK). The kernels leave their `float` denormals to the device exactly
/// where the profile takes `DriverDeviation::MetalFloatDenormals`.
#[test]
fn owned_dot_fixtures_answer_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let denormals = takes(&device, DriverDeviation::MetalFloatDenormals);
        let mut failures = Vec::new();
        for case_name in DOT_CASES {
            let case = public_case(case_name).unwrap();
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalFloatDenormals),
                denormals,
                "{case_name} on {name}: the kernel leaves its denormals to the device exactly \
                 where the profile takes the deviation"
            );
            let run = run_case(&device, &case).unwrap();
            let metal = metal_row(&case, METAL_ENVIRONMENT);
            assert_eq!(run.output.len(), metal.len(), "{case_name} on {name}");
            let differ = differing_of(&run.output, &metal, 8, &(0..8).collect::<Vec<_>>());
            eprintln!(
                "owned dot {case_name} on {name} (the device's denormals: {denormals}): {} of {} \
                 words differ from Metal",
                differ.len(),
                metal.len() / 4
            );
            if !differ.is_empty() {
                failures.push(format!(
                    "{case_name} on {name}: {} words differ from Metal (thread, word, device, \
                     Metal): {:x?}",
                    differ.len(),
                    &differ[..differ.len().min(24)]
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The minimum and maximum family.

/// The NaN model's minimum and maximum cases: Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` and
/// `median3` of `float` over NaN operands of every kind, and at zeros and denormals, without and
/// under fast math.
const MIN_MAX_CASES: [&str; 4] = [
    "nan-model-min-max-of-nan-operands",
    "nan-model-min-max-of-nan-operands-under-fast-math",
    "nan-model-min-max-at-zeros-and-denormals",
    "nan-model-min-max-at-zeros-and-denormals-under-fast-math",
];

/// Every word of [`MIN_MAX_CASES`] is the Metal row's of the same chip on every profile: the
/// device's `NMin`, `NMax` and `NClamp` where the profile takes `DriverDeviation::MetalMinMax`
/// (the kernels then require `VkProperty::MetalMinMax`), else the operand the module picks
/// itself (`math::metal_min_max`).
#[test]
fn minimums_and_maximums_answer_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let metals = takes(&device, DriverDeviation::MetalMinMax);
        let mut failures = Vec::new();
        for case_name in MIN_MAX_CASES {
            let case = public_case(case_name).unwrap();
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalMinMax),
                metals,
                "{case_name} on {name}: the kernel takes the device's NMin, NMax and NClamp \
                 exactly where the profile takes the deviation"
            );
            let run = run_case(&device, &case).unwrap();
            let metal = metal_row(&case, METAL_ENVIRONMENT);
            assert_eq!(run.output.len(), metal.len(), "{case_name} on {name}");
            let word = |bytes: &[u8], i: usize| {
                u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
            };
            let differ: Vec<(usize, u32, u32)> = (0..metal.len() / 4)
                .map(|i| (i, word(&run.output, i), word(&metal, i)))
                .filter(|(_, a, b)| a != b)
                .collect();
            eprintln!(
                "min/max {case_name} on {name} (the device's: {metals}): {} of {} words differ \
                 from Metal",
                differ.len(),
                metal.len() / 4
            );
            if !differ.is_empty() {
                failures.push(format!(
                    "{case_name} on {name}: words differ from Metal (word, device, Metal): {:x?}",
                    &differ[..differ.len().min(24)]
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The `half` arithmetic.

/// Cases whose every word is `half` arithmetic (with the narrowing of `float` operands and the
/// `half` fused multiply-add of the NaN model's): the NaN model's `half` arithmetic, without and
/// under fast math, and the `half` denormal results of `+`, `-` and `*`.
const HALF_ARITHMETIC_CASES: [&str; 3] = [
    "nan-model-float-to-half-narrowing-and-half-arithmetic",
    "nan-model-float-to-half-narrowing-and-half-arithmetic-under-fast-math",
    "half-denormal-results-of-arithmetic-and-conversions",
];

/// Every word of [`HALF_ARITHMETIC_CASES`] is the Metal row's of the same chip on every profile:
/// the device's 16-bit `OpFAdd`, `OpFSub` and `Fma(a, b, -0.0)`, kept apart, where the profile
/// takes `DriverDeviation::MetalHalfArithmetic` (the kernels then require
/// `VkProperty::MetalHalfArithmetic`), else the `float` result rounded to `half` by the module.
#[test]
fn half_arithmetic_answers_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let metals = takes(&device, DriverDeviation::MetalHalfArithmetic);
        let mut failures = Vec::new();
        for case_name in HALF_ARITHMETIC_CASES {
            let case = public_case(case_name).unwrap();
            assert_eq!(
                requires_property(&device, &case, VkProperty::MetalHalfArithmetic),
                metals,
                "{case_name} on {name}: the kernel takes the device's 16-bit arithmetic exactly \
                 where the profile takes the deviation"
            );
            let run = run_case(&device, &case).unwrap();
            let metal = metal_row(&case, METAL_ENVIRONMENT);
            assert_eq!(run.output.len(), metal.len(), "{case_name} on {name}");
            let word = |bytes: &[u8], i: usize| {
                u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
            };
            let differ: Vec<(usize, u32, u32)> = (0..metal.len() / 4)
                .map(|i| (i, word(&run.output, i), word(&metal, i)))
                .filter(|(_, a, b)| a != b)
                .collect();
            eprintln!(
                "half arithmetic {case_name} on {name} (the device's: {metals}): {} of {} words \
                 differ from Metal",
                differ.len(),
                metal.len() / 4
            );
            if !differ.is_empty() {
                failures.push(format!(
                    "{case_name} on {name}: words differ from Metal (word, device, Metal): {:x?}",
                    &differ[..differ.len().min(24)]
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The fragment simdgroups.

/// The side of the fragment lane draws' square `R32Uint` target.
const LANE_SIDE: usize = 32;

/// What a pixel no draw reaches holds: the clear value.
const UNDRAWN: u32 = u32::MAX;

/// The fragment lane draws, each a triangle list in clip space: one triangle over the whole
/// target, one triangle over part of it, and a 5x5 grid of quads of two triangles each. Each
/// vertex is at a pixel position that is a multiple of 1/16, so each clip coordinate is a multiple
/// of 1/256, exact in a `float`: Metal and the device rasterize the same triangles.
fn lane_draws() -> [(&'static str, Vec<[f32; 2]>); 3] {
    let at = |x: f32, y: f32| [x / 16.0 - 1.0, 1.0 - y / 16.0];
    let full = vec![[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];
    let partial = vec![at(1.25, 1.75), at(30.125, 9.875), at(4.75, 29.25)];
    let edges = [0.0, 6.375, 12.8125, 19.1875, 25.625, 32.0];
    let mut grid = Vec::new();
    for j in 0..5 {
        for i in 0..5 {
            let (x0, x1, y0, y1) = (edges[i], edges[i + 1], edges[j], edges[j + 1]);
            grid.extend([
                at(x0, y0),
                at(x1, y0),
                at(x0, y1),
                at(x1, y0),
                at(x1, y1),
                at(x0, y1),
            ]);
        }
    }
    [("full", full), ("partial", partial), ("grid", grid)]
}

/// Metal's lane of the pixel `(x, y)` of a primitive that covers its aligned 8x4 block whole:
/// the block's 2x2 quads in row-major order, each quad's pixels in Metal's quad order.
fn whole_block_lane(x: usize, y: usize) -> u32 {
    let (x, y) = (x as u32, y as u32);
    (x & 1) | (y & 1) << 1 | (x >> 1 & 3) << 2 | (y >> 1 & 1) << 4
}

/// Metal's lanes of the partial triangle, one string a row of the target: `.` where the triangle
/// does not reach, else the lane in base 32. Metal compiled the fixture's MSL at run time and drew
/// it on the M3 Max (macOS 26.7) and the M4 (macOS 27.0.1), the same lanes on both, for each of the
/// fixture's three forms of the lane.
const PARTIAL_LANES: [&str; LANE_SIDE] = [
    "................................",
    "................................",
    ".145............................",
    ".367abe.........................",
    "..ghklop89c.....................",
    "..ijmnqrabefijm.................",
    "..st0145opst0145op..............",
    "..uv2367qruv2367qruv23..........",
    "..89cdgh014589cd4589cdgh4.......",
    "..abefij2367abef67abefij67abe...",
    "..klopstghklopstklopst01ghklo...",
    "..mnqruvijmnqruvmnqruv23ijmn....",
    "...14589opst0145st014589st0.....",
    "...367abqruv2367uv2367abu.......",
    "...dghkl89cdghklcdghklop........",
    "...fijmnabefijmnefijmnq.........",
    "...pst01ghklopst4589c...........",
    "...ruv23ijmnqruv67ab............",
    "...589cd014589cdghk.............",
    "....abef2367abefij..............",
    "....ghkl014589cd................",
    "....ijmn2367abe.................",
    "....opstghklop..................",
    "....qruvijmn....................",
    "....st01cdg.....................",
    "....uv23ef......................",
    "....4589........................",
    ".....7a.........................",
    ".....l..........................",
    "................................",
    "................................",
    "................................",
];

/// Metal's lanes of the grid of quads, as [`PARTIAL_LANES`]: one simdgroup holds the quads of
/// several cells' triangles.
const GRID_LANES: [&str; LANE_SIDE] = [
    "014589cd89cdgdkl89cpghkls5ghklo9",
    "2367a3efabebifmnabarijmn67ijmnab",
    "ghklopstopst0l45odsh014l01st0d45",
    "ijm7qruvqjuv2n67efuj23mn23uvef67",
    "opst0145ghklo1s589cpgtkl01s90145",
    "qbuv236fijmnq367aberuvmn23ab2367",
    "014589cdklops901ghk589cdg1klopst",
    "2367ajefmnqr6b23ijq7abef23mnqru7",
    "cdghklopklolsp01ctg9cdg189cdghkl",
    "efifmnqrmnmnur23uvibef23abefibmn",
    "st0145894t89c1g5klolo5stopst0145",
    "uj2367abuvabe367mnqn67uvqruf2367",
    "89cdghkpst014d89014hklopst014589",
    "2367ab23efijmfqrabebijmnqnuv236j",
    "cdghk589st0h4l89odsh014589cdghop",
    "efij67abuvij6nabefuj236nabefinqr",
    "opspst01opst0t41opsp01454589klop",
    "qrqruv23qruv2v23qrur2v6767ajmnqr",
    "0589cdgl89cdg5kl89c1ghklcd014589",
    "2367abefijmnq3uvuv2f67abujmnqruv",
    "4589c5ghcdgdkhopcdgdklopcdghklod",
    "67ab67ijefefmjqrebifmnq3efijmnef",
    "klo9st01sl014p89st0h4589st014h89",
    "mnabuv23mn236ravuv2j67abuv23ijab",
    "0h4589clopst0945klotopst014h89cd",
    "ij67abmnqruv2f67mnqnqruv23ijabef",
    "01458d014589cpghst0h4589k5opst09",
    "2367ef2367anerijuf2j67ab67qruvab",
    "ghk9opst89cdgdklcdgdklohst014t89",
    "ijabqruvabefifmjefifmnijuv23uvab",
    "sd014501opst0l45st0l4p89cdg1klop",
    "ef236727qruv2r67uv2nqrabef23mnqr",
];

/// Metal's lanes of `draw`, row by row ([`UNDRAWN`] where the draw does not reach).
fn metal_lanes(draw: &str) -> Vec<u32> {
    let rows = match draw {
        "partial" => PARTIAL_LANES,
        "grid" => GRID_LANES,
        _ => {
            return (0..LANE_SIDE * LANE_SIDE)
                .map(|i| whole_block_lane(i % LANE_SIDE, i / LANE_SIDE))
                .collect()
        }
    };
    rows.iter()
        .flat_map(|row| row.chars())
        .map(|c| c.to_digit(32).unwrap_or(UNDRAWN))
        .collect()
}

/// The pixels of `fragment` of the owned fixture `gsm2_fragment_lanes` drawn over `vertices` (a
/// triangle list) into a 32x32 target of `format` (`R32Uint`, or `RGBA32Uint` for
/// `fragment_lane_shuffles`) cleared to [`UNDRAWN`], row by row, each pixel its `components`
/// words; or why the pipeline state or the draw was refused.
fn lane_draw(
    device: &Device,
    fragment: &str,
    vertices: &[[f32; 2]],
    (format, components): (PixelFormat, u64),
) -> Result<Vec<u32>, RuntimeError> {
    let bitcode = llvm_assemble(&fixture("gsm2_fragment_lanes.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = |name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("fragment_lanes_vertex")));
    descriptor.fragment = Some(StageSource::Function(function(fragment)));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
    let pipeline = device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())?;
    let side = LANE_SIDE as u64;
    let mut target = TextureDescriptor::texture_2d(format, side, side, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    let undrawn = f64::from(UNDRAWN);
    attachment.clear_color = ClearColor::new(undrawn, undrawn, undrawn, undrawn);
    attachment.store_action = StoreAction::Store;
    let row = 4 * components * side;
    let length = row * side;
    let readback = buffer_with(device, &vec![0; length as usize]).unwrap();
    let positions: Vec<u8> = vertices
        .iter()
        .flatten()
        .flat_map(|c| c.to_le_bytes())
        .collect();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&pipeline)?;
    encoder.set_vertex_bytes(&positions, 0)?;
    encoder.draw_primitives(PrimitiveType::Triangle, 0, vertices.len() as u32, 1, 0)?;
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &readback,
        0,
        row,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    Ok(read_buffer(&readback, 0, length)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

/// The pixels of `drawn` that differ from `metal`: `(x, y, drawn, Metal)`.
fn differing_lanes(drawn: &[u32], metal: &[u32]) -> Vec<(usize, usize, u32, u32)> {
    (0..metal.len())
        .filter(|&i| drawn.get(i) != metal.get(i))
        .map(|i| {
            let d = drawn.get(i).copied().unwrap_or(UNDRAWN);
            (i % LANE_SIDE, i / LANE_SIDE, d, metal[i])
        })
        .collect()
}

/// Why the lanes `drawn` by a device whose fragment subgroups are not Metal's simdgroups are not
/// a grouping Metal could give for the draw `metal` holds Metal's lanes of: a pixel covered by
/// one and not the other, or a lane of 32 or more.
fn device_lanes(drawn: &[u32], metal: &[u32]) -> Option<String> {
    let wrong: Vec<(usize, usize, u32)> = (0..metal.len())
        .filter(|&i| {
            let d = drawn.get(i).copied().unwrap_or(UNDRAWN);
            (d == UNDRAWN) != (metal[i] == UNDRAWN) || (d != UNDRAWN && d >= 32)
        })
        .map(|i| {
            (
                i % LANE_SIDE,
                i / LANE_SIDE,
                drawn.get(i).copied().unwrap_or(UNDRAWN),
            )
        })
        .collect();
    (!wrong.is_empty()).then(|| {
        format!(
            "pixels uncovered as Metal's are not, or lanes of 32 or more (x, y, drawn): {:?}",
            &wrong[..wrong.len().min(24)]
        )
    })
}

/// Why `error` is not the refusal a device without Metal's fragment simdgroups (`metals` false)
/// gives a fragment lane where its fragment stage runs no 32-lane subgroups: `SimdWidth`, needing
/// `VkProperty::MetalFragmentSimdgroups` or what the stage lacks for 32 lanes, or, on a stage
/// without the subgroup tier, `SubgroupOperations`.
fn lane_refusal(error: &RuntimeError, metals: bool) -> Option<String> {
    let property = vec![Capability::Property(VkProperty::MetalFragmentSimdgroups)];
    match error {
        RuntimeError::Compile(CompileError::Unsupported(u)) | RuntimeError::Unsupported(u)
            if !metals =>
        {
            let width = u.feature == Feature::SimdWidth
                && matches!(&u.needs, Needs::AnyOf(alternatives) if alternatives.first() == Some(&property));
            (!width && u.feature != Feature::SubgroupOperations).then(|| format!("{error}"))
        }
        _ => Some(format!("{error}")),
    }
}

/// The owned fixture `gsm2_fragment_lanes` over the three lane draws:
/// - Where the profile takes `DriverDeviation::MetalFragmentSimdgroups`, each of the three forms
///   of a fragment's simdgroup lane (a `uint` and a `ushort` parameter, a program-scope `uint`)
///   draws Metal's lanes of the same draws on every pixel, those of a primitive covering an
///   aligned 8x4 block whole and those the rasterizer packs across blocks and primitives.
/// - Elsewhere, where the device's fragment stage runs 32-lane subgroups, each draws the lanes of
///   the device's own grouping: every pixel Metal covers, each lane below 32. Where it
///   runs none, each pipeline state is refused ([`lane_refusal`]).
/// - `fragment_lane_shuffles` writes the lane beside a shuffle of the lane from itself, the first
///   active lane by the ballot's bits, and a shuffle of the lane from that first lane. Wherever it
///   is made, each pixel's own shuffle is its lane, the first lane's shuffle is the first lane,
///   and the first lane is at most its lane: the lane, the ballot and the shuffles name the same
///   lanes of the same subgroup. Its lanes are the lane forms' as above.
/// - A `[[threads_per_simdgroup]]` parameter draws 32 on every device.
#[test]
fn fragment_simdgroup_lanes_are_metal_s() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let mut failures = Vec::new();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let metals = takes(&device, DriverDeviation::MetalFragmentSimdgroups);
        let r32 = (PixelFormat::R32Uint, 1);
        let lanes_verdict = |drawn: &[u32], metal: &[u32], what: &str| {
            let differ = differing_lanes(drawn, metal);
            eprintln!(
                "fragment lanes {what} on {name} (Metal's simdgroups: {metals}): {} of {} pixels \
                 differ from Metal",
                differ.len(),
                metal.len()
            );
            if !metals {
                device_lanes(drawn, metal)
            } else if !differ.is_empty() {
                Some(format!(
                    "pixels differ from Metal (x, y, drawn, Metal): {:?}",
                    &differ[..differ.len().min(24)]
                ))
            } else {
                None
            }
        };
        for (draw, vertices) in lane_draws() {
            let metal = metal_lanes(draw);
            for fragment in [
                "fragment_lane",
                "fragment_lane_ushort",
                "fragment_global_lane",
            ] {
                let verdict = match lane_draw(&device, fragment, &vertices, r32) {
                    Ok(drawn) => lanes_verdict(&drawn, &metal, &format!("{draw} {fragment}")),
                    Err(error) => {
                        eprintln!("fragment lanes {draw} {fragment} on {name}: {error}");
                        lane_refusal(&error, metals)
                    }
                };
                if let Some(why) = verdict {
                    failures.push(format!("{draw} {fragment} on {name}: {why}"));
                }
            }
            let shuffles = "fragment_lane_shuffles";
            let verdict =
                match lane_draw(&device, shuffles, &vertices, (PixelFormat::RGBA32Uint, 4)) {
                    Ok(words) => {
                        let pixels: Vec<&[u32]> = words
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|p| p.as_slice())
                            .collect();
                        let broken: Vec<(usize, usize, &[u32])> = (0..pixels.len())
                            .filter(|&i| {
                                let [lane, own, first, from_first] = pixels[i] else {
                                    return true;
                                };
                                *lane != UNDRAWN
                                    && (own != lane || from_first != first || first > lane)
                            })
                            .map(|i| (i % LANE_SIDE, i / LANE_SIDE, pixels[i]))
                            .collect();
                        eprintln!(
                            "fragment lanes {draw} {shuffles} on {name}: {} pixels break the \
                         shuffles' relations",
                            broken.len()
                        );
                        let lanes: Vec<u32> = pixels.iter().map(|p| p[0]).collect();
                        if !broken.is_empty() {
                            Some(format!(
                                "the lane, the ballot and the shuffles disagree (x, y, [lane, own \
                             shuffle, first, first's shuffle]): {:?}",
                                &broken[..broken.len().min(24)]
                            ))
                        } else {
                            lanes_verdict(&lanes, &metal, &format!("{draw} {shuffles}"))
                        }
                    }
                    Err(error) => {
                        eprintln!("fragment lanes {draw} {shuffles} on {name}: {error}");
                        lane_refusal(&error, metals)
                    }
                };
            if let Some(why) = verdict {
                failures.push(format!("{draw} {shuffles} on {name}: {why}"));
            }
            match lane_draw(&device, "fragment_simd_width", &vertices, r32) {
                Ok(drawn) => {
                    let widths: std::collections::BTreeSet<u32> =
                        drawn.iter().copied().filter(|&w| w != UNDRAWN).collect();
                    if widths != [32].into() {
                        failures.push(format!(
                            "{draw} fragment_simd_width on {name}: widths {widths:?}"
                        ));
                    }
                }
                Err(error) => {
                    failures.push(format!("{draw} fragment_simd_width on {name}: {error}"))
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Vertex simdgroups.

/// The owned fixtures of the vertex simdgroup cases: the `rvc_simd_vertex` (a sum, a shuffle from
/// lane 5, a shuffle by XOR and a ballot) and `rvc_simd_vertex_fill` (a shuffle-and-fill down by 3
/// with an explicit width of 32, which reads the simdgroup lane), and the
/// `gsm2_vertex_votes_shifts` (votes, and shuffles up and down and a fill up, over 64 points and
/// over 40, whose second simdgroup is partial).
const VERTEX_SIMDGROUP_FIXTURES: [&str; 3] = [
    "rvc_simd_vertex.ll",
    "rvc_simd_vertex_fill.ll",
    "gsm2_vertex_votes_shifts.ll",
];

/// The committed cases of the owned fixture `name`, each with its Metal row's output in the
/// lane's environment (the M3 Max's; the M4 drew the same words).
fn fixture_cases(name: &str) -> Vec<(AuthoredCase, Vec<u8>)> {
    let source = format!("{}/engine/{name}", crate::source::PUBLIC_FIXTURE_DIR);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let mut out = Vec::new();
    for shard in 0..64 {
        for case in store.read_case_shard(shard).unwrap() {
            if case.public_source.as_deref() == Some(source.as_str()) {
                let metal = super::metal_output(&store, &case)
                    .unwrap_or_else(|e| panic!("{}: {e}", case.name));
                out.push((case, metal));
            }
        }
    }
    out
}

/// The rasterization-disabled render pipeline descriptor of `case`'s entry in the owned fixture
/// `name` on `device`.
fn vertex_descriptor(device: &Device, name: &str, case: &AuthoredCase) -> RenderPipelineDescriptor {
    let bitcode = llvm_assemble(&fixture(name)).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(&case.entry).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    descriptor
}

/// The output of `case`'s draw of rasterization-disabled points on `device`, from the pipeline
/// state of `descriptor`, each of its buffers bound to the vertex stage at its binding; or why
/// the state was refused.
fn vertex_points(
    device: &Device,
    descriptor: &RenderPipelineDescriptor,
    case: &AuthoredCase,
) -> Result<Vec<u8>, RuntimeError> {
    let draw = case.draw.as_ref().expect("a vertex case's draw");
    assert_eq!(draw.primitive, Primitive::Point, "{}", case.name);
    let state = device.new_render_pipeline_state(descriptor, &PipelineOptions::default())?;
    let buffers: Vec<(u32, Buffer)> = case
        .buffers
        .iter()
        .map(|resource| {
            let bytes = resource
                .bytes_b64
                .as_deref()
                .or(resource.initial_bytes_b64.as_deref())
                .expect("the buffer's bytes");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(bytes)
                .unwrap();
            (resource.binding, buffer_with(device, &bytes).unwrap())
        })
        .collect();
    let OutputSelection::Buffer {
        binding,
        offset,
        length,
    } = case.output
    else {
        panic!("{}: a buffer output", case.name)
    };
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    for (index, buffer) in &buffers {
        encoder.set_vertex_buffer(Some(buffer), 0, *index).unwrap();
    }
    encoder
        .draw_primitives(
            PrimitiveType::Point,
            draw.vertex_start,
            draw.vertex_count,
            draw.instance_count,
            draw.base_instance,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "{}", case.name);
    let (_, out) = buffers
        .iter()
        .find(|(index, _)| *index == binding)
        .expect("the output buffer");
    Ok(read_buffer(out, offset, length).unwrap())
}

/// The vertex simdgroup cases ([`VERTEX_SIMDGROUP_FIXTURES`]) draw Metal's words where the
/// profile takes `DriverDeviation::MetalVertexSimdgroups`: MoltenVK on the M3 Max and the M4, whose
/// `subgroupSupportedStages` leaves the vertex stage out, but which hands the vertex function to
/// Metal, which runs it in simdgroups of 32 consecutive vertices, the active ones the low lanes.
/// There each pipeline requires `VkProperty::MetalVertexSimdgroups` in place of the vertex stage,
/// the fill's lane is the count of the active lanes below the invocation's, the runtime announces
/// `Degradation::UnadvertisedVertexSimdgroups` once, and the validation layer's report of the stage
/// (`VUID-RuntimeSpirv-None-06343`) passes only after it. Elsewhere a device whose vertex stage has
/// subgroups draws its own grouping (C3 compares it: RADV and NVIDIA match, ANV regroups), which
/// the test prints, and one without them refuses each state, typed (`SubgroupOperations`).
#[test]
fn vertex_simdgroups_are_metal_s() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let compile = device.compile_device();
        let profile = compile.profile();
        let takes = profile.takes_driver_deviation(DriverDeviation::MetalVertexSimdgroups);
        let unadvertised = takes
            && !profile
                .enabled()
                .properties
                .contains(&VkProperty::SubgroupStage(Stage::Vertex));
        let mut failures = Vec::new();
        for fixture_name in VERTEX_SIMDGROUP_FIXTURES {
            let cases = fixture_cases(fixture_name);
            assert!(!cases.is_empty(), "{fixture_name}: no committed case");
            for (case, metal) in cases {
                let descriptor = vertex_descriptor(device, fixture_name, &case);
                let drawn = match vertex_points(device, &descriptor, &case) {
                    Ok(drawn) => drawn,
                    Err(error) => {
                        let RuntimeError::Unsupported(u) = &error else {
                            panic!("{} on {name}: {error}", case.name);
                        };
                        assert_eq!(u.feature, Feature::SubgroupOperations, "{u:?}");
                        assert!(
                            !profile.subgroups().vertex.supported,
                            "{} on {name}: refused with vertex subgroups: {u:?}",
                            case.name
                        );
                        eprintln!("{} on {name}: refused, typed: {u:?}", case.name);
                        continue;
                    }
                };
                let compiled = compile
                    .compile_render_pipeline(&descriptor, &Budget::standard())
                    .unwrap();
                assert_eq!(
                    compiled
                        .requirements
                        .properties
                        .contains(&VkProperty::MetalVertexSimdgroups),
                    unadvertised,
                    "{} on {name}: the pipeline requires the deviation's property exactly where \
                     the profile takes it and the device does not list the vertex stage",
                    case.name
                );
                let differ = differing_of(&drawn, &metal, 1, &[0]);
                eprintln!(
                    "{} on {name} (the deviation's: {takes}, unadvertised stage: \
                     {unadvertised}): {} of {} words differ from Metal",
                    case.name,
                    differ.len(),
                    metal.len() / 4
                );
                if takes && !differ.is_empty() {
                    failures.push(format!(
                        "{} on {name}: {} words differ from Metal (word, 0, device, Metal): {:x?}",
                        case.name,
                        differ.len(),
                        &differ[..differ.len().min(24)]
                    ));
                }
            }
        }
        let notices = notice_counts(&messages.take_notices());
        assert_eq!(
            notices.get(&Degradation::UnadvertisedVertexSimdgroups),
            unadvertised.then_some(&1),
            "{notices:?}"
        );
        assert!(failures.is_empty(), "{failures:#?}");
    });
}
