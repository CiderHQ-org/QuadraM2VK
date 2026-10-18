//! The depth a fragment shader sees against the depth its device stores (refinements 1 and 2): the
//! first step of the in-shader depth test.
//!
//! The runtime can move a draw's depth test into the shader, inside the pixel interlock, where it
//! compares `clamp(FragCoord.z, the viewport's depth range)` with the depth plane. That is exact on
//! a device only where the clamped `FragCoord.z` is, bit for bit, the depth the device's own
//! fixed-function test stores for the same draw, in the form the product ships. This step measures
//! that, per case and per form, on the device a venue run selects.
//!
//! Each case draws its triangles over a `SIDE`² target, at one sample and at `MSAA` samples, in
//! four render pass instances, each its own command recorded in one command buffer:
//!
//! - **plain**: an `R32_SFLOAT` color and a depth attachment (`D32_SFLOAT_S8_UINT` where the device
//!   samples and renders it at both sample counts, else `D32_SFLOAT`), cleared to 1, the depth test
//!   `ALWAYS` with writes: the attachment then holds the device's own stored depth, which a compute
//!   dispatch reads back through a depth-aspect view, sample by sample;
//! - **out**: the color alone, the depth attachment out of the subpass (the form);
//! - **kept_off**: the depth attachment in the subpass with its test and writes off (item 7's
//!   second form, refinement 1);
//! - **kept_always**: the same with the test on, `ALWAYS`, and writes off.
//!
//! In every instance the fragment shader writes `FragCoord.z` into a storage buffer at the pixel
//! (and, at `MSAA` samples, at its `SampleId`, which makes it run per sample) and to the color. The
//! rasterizer state is the case's: the viewport's depth range, depth clamp (Metal's clamp mode),
//! and the depth bias's constant, slope and clamp factors, recorded as the product records them
//! (`metal::state::depth_bias`: Metal's factors, in the device's unit for a float depth format).
//!
//! The cases are our probe's (`dzprobe.swift`), whose Metal answers were recorded: flat,
//! sloped, steep and perspective triangles; constant, negative, slope-scaled, clamped and
//! perspective bias; depth clamp with depth beyond both planes, and clipped; narrowed, reversed
//! and narrowed-with-clamp viewport ranges. Each form's report counts, over the samples the plain
//! instance covered, those whose clamped `FragCoord.z` equals the stored depth bit for bit, those
//! whose raw `FragCoord.z` does, the largest distance in units in the last place, and the first
//! differences; and whether the form covered the same samples.
//!
//! The clamp the reports apply is Metal's: to `[0, 1]` in clip mode, to the viewport's range in
//! clamp mode. A device that enabled `VK_EXT_depth_range_unrestricted` stores a depth that a bias
//! takes past `[0, 1]` in clip mode unclamped (RADV and NVIDIA), so where the device enabled
//! `depthClamp`, `VK_EXT_depth_clip_enable`'s `depthClipEnable` and `VK_EXT_depth_clamp_control`'s
//! `depthClampControl`, every case runs a second time with the ordinary draw's candidate fix
//! (`ordinary_clamp`): in clip mode, depth clamp on with the user-defined range `[0, 1]` and
//! clipping kept. Two cases of a flat triangle through a 0.25-to-0.75 range in clip mode take a
//! constant bias past the range and past 1, which Metal stores clamped to `[0, 1]` alone.
//!
//! At `MSAA` samples the plain instance's shader also computes per-sample depth candidates
//! ([`CANDIDATES`]: from `FragCoord.z` and its derivatives, and from a varying of the vertex's
//! depth without perspective interpolated at the sample), for a device whose `FragCoord.z` is the
//! pixel centre's at every sample (MoltenVK, ANV, NVIDIA): each is compared, as a form is, with
//! the depth the device stores at each sample (the computed per-sample depth).
//!
//! Then [`stored`] reads the stored depth of four cases at one sample three ways (the readback's
//! fetch, a copy, and a draw's own `EQUAL` test), three times each.

use super::super::pipeline::create_shader_module;
use super::{result_name, Object, Owned, VulkanContext};

mod stored;
use ash::vk;
use metal2vulkan::device::{VkExtension, VkFeature};
use serde::Serialize;

/// The target's side, as the Metal probe's.
const SIDE: u32 = 64;

/// The multisampled cases' samples.
const MSAA: u32 = 4;

/// The bits a storage buffer holds where no fragment wrote: a NaN no depth is.
const UNWRITTEN: u32 = 0xFFFF_FFFF;

/// One draw's state: its triangles' clip-space vertices, Metal's depth bias (constant, slope
/// scale, clamp), its depth clip mode (`clamp`: Metal's `MTLDepthClipModeClamp`), and its
/// viewport's depth range (near, far).
struct Case {
    name: &'static str,
    vertices: &'static [[f32; 4]],
    bias: [f32; 3],
    clamp: bool,
    range: [f32; 2],
}

const FULL: &[[f32; 4]] = &[
    [-1.0, -1.0, 0.3, 1.0],
    [3.0, -1.0, 0.3, 1.0],
    [-1.0, 3.0, 0.3, 1.0],
];
const SLOPED: &[[f32; 4]] = &[
    [-1.0, -1.0, 0.1, 1.0],
    [3.0, -1.0, 0.7, 1.0],
    [-1.0, 3.0, 0.9, 1.0],
];
const STEEP: &[[f32; 4]] = &[
    [-1.0, -1.0, 0.0, 1.0],
    [1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, 0.5, 1.0],
    [1.0, -1.0, 1.0, 1.0],
    [1.0, 1.0, 0.02, 1.0],
    [-1.0, 1.0, 0.5, 1.0],
];
const PERSPECTIVE: &[[f32; 4]] = &[
    [-0.5, -0.5, 0.05, 0.5],
    [3.0, -1.0, 0.6, 1.0],
    [-2.0, 6.0, 3.0, 4.0],
];
const OVER: &[[f32; 4]] = &[
    [-1.0, -1.0, -0.5, 1.0],
    [3.0, -1.0, 1.5, 1.0],
    [-1.0, 3.0, 0.5, 1.0],
];

const fn case(name: &'static str, vertices: &'static [[f32; 4]]) -> Case {
    Case {
        name,
        vertices,
        bias: [0.0; 3],
        clamp: false,
        range: [0.0, 1.0],
    }
}

const fn biased(name: &'static str, vertices: &'static [[f32; 4]], bias: [f32; 3]) -> Case {
    Case {
        bias,
        ..case(name, vertices)
    }
}

const fn ranged(
    name: &'static str,
    vertices: &'static [[f32; 4]],
    clamp: bool,
    range: [f32; 2],
) -> Case {
    Case {
        clamp,
        range,
        ..case(name, vertices)
    }
}

/// Our probe's cases but its two returned-depth ones, which test no `FragCoord.z`.
const CASES: &[Case] = &[
    case("flat", FULL),
    case("sloped", SLOPED),
    case("steep", STEEP),
    case("persp", PERSPECTIVE),
    biased("bias_const", SLOPED, [1000.0, 0.0, 0.0]),
    biased("bias_neg", SLOPED, [-1000.0, 0.0, 0.0]),
    biased("bias_slope", SLOPED, [0.0, 2.0, 0.0]),
    biased("bias_steep", STEEP, [0.0, 1.5, 0.0]),
    biased("bias_both_clamp", SLOPED, [500.0, 3.0, 0.0001]),
    biased("bias_persp", PERSPECTIVE, [64.0, 1.0, 0.0]),
    ranged("clamp", OVER, true, [0.0, 1.0]),
    ranged("clip", OVER, false, [0.0, 1.0]),
    ranged("viewport", SLOPED, false, [0.25, 0.75]),
    ranged("viewport_rev", SLOPED, false, [0.9, 0.1]),
    ranged("viewport_clamp", OVER, true, [0.25, 0.75]),
    Case {
        bias: [1.5e7, 0.0, 0.0],
        ..ranged("bias_out_range", FULL, false, [0.25, 0.75])
    },
    Case {
        bias: [4.0e7, 0.0, 0.0],
        ..ranged("bias_out_one", FULL, false, [0.25, 0.75])
    },
];

/// `VkPipelineViewportDepthClampControlCreateInfoEXT`, which `ash 0.38` predates, laid out as the
/// registry defines it (extension number 583), as the runtime's `pipeline::graphics` lays it out.
#[repr(C)]
struct DepthClampControlCreateInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    /// `VkDepthClampModeEXT`: `VK_DEPTH_CLAMP_MODE_USER_DEFINED_RANGE_EXT` is 1.
    depth_clamp_mode: i32,
    /// `const VkDepthClampRangeEXT *`: `minDepthClamp`, `maxDepthClamp`.
    depth_clamp_range: *const [f32; 2],
}

// SAFETY: the layout is the registry's `VkPipelineViewportDepthClampControlCreateInfoEXT`, which
// begins with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl vk::TaggedStructure for DepthClampControlCreateInfo {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_582_001);
}
// SAFETY: the registry lists `VkPipelineViewportStateCreateInfo` in the structure's
// `structextends`.
unsafe impl vk::ExtendsPipelineViewportStateCreateInfo for DepthClampControlCreateInfo {}

/// A render pass instance's form (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    Plain,
    Out,
    KeptOff,
    KeptAlways,
    /// [`stored`]'s second draw: the same triangles under `EQUAL` without writes, in a pass that
    /// loads the plain draw's stored depth.
    Tested,
}

const FORMS: [Form; 4] = [Form::Plain, Form::Out, Form::KeptOff, Form::KeptAlways];

impl Form {
    fn name(self) -> &'static str {
        match self {
            Form::Plain => "plain",
            Form::Out => "out",
            Form::KeptOff => "kept_off",
            Form::KeptAlways => "kept_always",
            Form::Tested => "tested",
        }
    }

    /// Whether the instance's subpass holds the depth attachment.
    fn depth(self) -> bool {
        self != Form::Out
    }
}

/// What the device answered.
#[derive(Debug, Default, Serialize)]
pub(super) struct DepthSource {
    /// The depth attachment's format.
    format: Option<String>,
    /// Whether the device enabled `VK_EXT_depth_range_unrestricted`.
    depth_range_unrestricted: bool,
    /// Whether the cases ran a second time with the ordinary draw's clip-mode clamp (the module
    /// docs).
    ordinary_clamp: bool,
    /// Whether the multisampled cases ran (the device renders and samples the format at `MSAA`
    /// samples, and shades per sample).
    multisampled: bool,
    /// Per case and sample count.
    cases: Vec<CaseReport>,
    /// Per form and sample count: whether every case's clamped `FragCoord.z` equals the stored
    /// depth at every sample, with the same coverage.
    verdicts: Vec<Verdict>,
    /// The stored depth of four cases read three ways, three times each.
    stored_reads: Vec<stored::StoredRead>,
    /// What stopped the step.
    error: Option<String>,
    /// The validation layer's messages during the step.
    pub(super) validation_messages: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    case: &'static str,
    samples: u32,
    /// Whether the run had the ordinary draw's clip-mode clamp (the module docs).
    ordinary_clamp: bool,
    forms: Vec<FormReport>,
    /// At more than one sample, each of [`CANDIDATES`] the plain instance's shader computed,
    /// against its stored depth.
    candidates: Vec<FormReport>,
    /// What stopped the case (a feature the device lacks, or a failure).
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct FormReport {
    form: &'static str,
    /// The samples the plain instance covered.
    covered: u32,
    /// Whether this form covered exactly those.
    same_coverage: bool,
    /// Covered samples whose `FragCoord.z`, clamped as Metal clamps it, is the stored depth bit
    /// for bit.
    equal_clamped: u32,
    /// Covered samples whose raw `FragCoord.z` is.
    equal_raw: u32,
    /// The largest distance between the clamped `FragCoord.z` and the stored depth, in units in
    /// the last place (0 when all are equal).
    max_ulp: u64,
    /// The first differing samples: index, `FragCoord.z`'s bits, the stored depth's bits.
    first_differences: Vec<[u32; 3]>,
}

#[derive(Debug, Serialize)]
struct Verdict {
    form: &'static str,
    samples: u32,
    ordinary_clamp: bool,
    /// Cases run, and those exact at every sample.
    cases: u32,
    exact: u32,
}

pub(super) fn measure(context: &VulkanContext) -> DepthSource {
    let mut out = DepthSource::default();
    if let Err(error) = measure_into(context, &mut out) {
        out.error = Some(error);
    }
    for (form, ordinary_clamp, samples) in FORMS
        .into_iter()
        .flat_map(|form| [false, true].map(|clamped| (form, clamped)))
        .flat_map(|(form, clamped)| [1, MSAA].map(|samples| (form, clamped, samples)))
    {
        let runs = out.cases.iter().filter(|c| {
            c.samples == samples && c.ordinary_clamp == ordinary_clamp && c.error.is_none()
        });
        let (mut cases, mut exact) = (0, 0);
        for report in runs {
            cases += 1;
            let form_report = report.forms.iter().find(|f| f.form == form.name());
            if form_report.is_some_and(|f| f.same_coverage && f.equal_clamped == f.covered) {
                exact += 1;
            }
        }
        if cases > 0 {
            out.verdicts.push(Verdict {
                form: form.name(),
                samples,
                ordinary_clamp,
                cases,
                exact,
            });
        }
    }
    for (name, ordinary_clamp) in CANDIDATES
        .into_iter()
        .flat_map(|name| [false, true].map(|clamped| (name, clamped)))
    {
        let runs = out.cases.iter().filter(|c| {
            c.samples == MSAA && c.ordinary_clamp == ordinary_clamp && c.error.is_none()
        });
        let (mut cases, mut exact) = (0, 0);
        for report in runs {
            cases += 1;
            let found = report.candidates.iter().find(|f| f.form == name);
            if found.is_some_and(|f| f.same_coverage && f.equal_clamped == f.covered) {
                exact += 1;
            }
        }
        if cases > 0 {
            out.verdicts.push(Verdict {
                form: name,
                samples: MSAA,
                ordinary_clamp,
                cases,
                exact,
            });
        }
    }
    out.validation_messages = context.take_messages();
    out
}

pub(super) fn enabled(context: &VulkanContext, feature: VkFeature) -> bool {
    context.enabled.features.contains(&feature)
}

/// The depth format, and whether it serves the multisampled cases.
fn depth_format(context: &VulkanContext) -> Result<(vk::Format, bool), String> {
    let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    let counts = |format: vk::Format| -> Option<vk::SampleCountFlags> {
        // SAFETY: the context's physical device belongs to its live instance.
        unsafe {
            context
                .instance
                .get_physical_device_image_format_properties(
                    context.physical,
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    usage,
                    vk::ImageCreateFlags::empty(),
                )
        }
        .ok()
        .map(|properties| properties.sample_counts)
    };
    // SAFETY: as above.
    let limits = unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical)
    }
    .limits;
    let msaa = vk::SampleCountFlags::from_raw(MSAA);
    let multisampled = enabled(context, VkFeature::SampleRateShading)
        && limits.framebuffer_color_sample_counts.contains(msaa)
        && limits.framebuffer_depth_sample_counts.contains(msaa)
        && limits.sampled_image_depth_sample_counts.contains(msaa);
    for format in [vk::Format::D32_SFLOAT_S8_UINT, vk::Format::D32_SFLOAT] {
        if let Some(found) = counts(format) {
            return Ok((format, multisampled && found.contains(msaa)));
        }
    }
    Err("the device renders and samples no 32-bit float depth format".into())
}

fn measure_into(context: &VulkanContext, out: &mut DepthSource) -> Result<(), String> {
    if !enabled(context, VkFeature::FragmentStoresAndAtomics) {
        return Err("fragmentStoresAndAtomics is not enabled".into());
    }
    let (format, multisampled) = depth_format(context)?;
    out.format = Some(
        match format {
            vk::Format::D32_SFLOAT_S8_UINT => "D32_SFLOAT_S8_UINT",
            _ => "D32_SFLOAT",
        }
        .to_string(),
    );
    out.multisampled = multisampled;
    out.depth_range_unrestricted = context
        .enabled
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted);
    out.ordinary_clamp = [
        VkFeature::DepthClamp,
        VkFeature::DepthClipEnable,
        VkFeature::DepthClampControl,
    ]
    .into_iter()
    .all(|feature| enabled(context, feature));
    let shaders = Shaders::assemble(multisampled)?;
    let sample_counts: &[u32] = if multisampled { &[1, MSAA] } else { &[1] };
    let clamps: &[bool] = if out.ordinary_clamp {
        &[false, true]
    } else {
        &[false]
    };
    for (&ordinary_clamp, &samples) in clamps
        .iter()
        .flat_map(|clamp| sample_counts.iter().map(move |samples| (clamp, samples)))
    {
        for case in CASES {
            let mut report = CaseReport {
                case: case.name,
                samples,
                ordinary_clamp,
                forms: Vec::new(),
                candidates: Vec::new(),
                error: None,
            };
            let lacking = [
                (case.clamp, VkFeature::DepthClamp, "depthClamp"),
                (
                    case.bias[2] != 0.0,
                    VkFeature::DepthBiasClamp,
                    "depthBiasClamp",
                ),
            ]
            .into_iter()
            .find(|&(needed, feature, _)| needed && !enabled(context, feature));
            match lacking {
                Some((_, _, name)) => report.error = Some(format!("{name} is not enabled")),
                None => {
                    match run_case(context, &shaders, (format, samples, ordinary_clamp), case) {
                        Ok((forms, candidates)) => {
                            report.forms = forms;
                            report.candidates = candidates;
                        }
                        Err(error) => report.error = Some(error),
                    }
                }
            }
            out.cases.push(report);
        }
    }
    out.stored_reads = stored::measure(context, &shaders, format);
    Ok(())
}

/// The step's SPIR-V: the vertex shader, and per sample count the fragment shader and the
/// readback's compute shader.
struct Shaders {
    vertex: Vec<u8>,
    fragment: Vec<(u32, Vec<u8>)>,
    readback: Vec<(u32, Vec<u8>)>,
}

impl Shaders {
    fn assemble(multisampled: bool) -> Result<Shaders, String> {
        let assemble = |text: &str, label: &str| {
            crate::spirv_tools::assemble_validated(text, label).map_err(String::from)
        };
        let counts: &[u32] = if multisampled { &[1, MSAA] } else { &[1] };
        let mut shaders = Shaders {
            vertex: assemble(VERTEX, "depth-source-vertex")?,
            fragment: Vec::new(),
            readback: Vec::new(),
        };
        for &samples in counts {
            shaders.fragment.push((
                samples,
                assemble(
                    &fragment(samples),
                    &format!("depth-source-fragment-{samples}"),
                )?,
            ));
            shaders.readback.push((
                samples,
                assemble(
                    &readback(samples),
                    &format!("depth-source-readback-{samples}"),
                )?,
            ));
        }
        Ok(shaders)
    }

    fn of(list: &[(u32, Vec<u8>)], samples: u32) -> &[u8] {
        &list
            .iter()
            .find(|(s, _)| *s == samples)
            .expect("assembled for every sample count run")
            .1
    }
}

/// The vertex shader: location 0 is the clip-space position. It also writes, without
/// perspective, the position's window depth by the viewport's depth range (specialization
/// constants 0 and 1, near and far) in four arithmetic orders, for the per-sample candidates:
/// location 0 `z / w · (far - near) + near`, 1 `fma(z / w, far - near, near)`, 2
/// `fma(z · (1 / w), far - near, near)`, 3 `z · (1 / w) · (far - near) + near`.
const VERTEX: &str = r#"
               OpCapability Shader
       %glsl = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %pos %out %w0 %w1 %w2 %w3
               OpDecorate %pos Location 0
               OpMemberDecorate %PV 0 BuiltIn Position
               OpDecorate %PV Block
               OpDecorate %w0 Location 0
               OpDecorate %w0 NoPerspective
               OpDecorate %w1 Location 1
               OpDecorate %w1 NoPerspective
               OpDecorate %w2 Location 2
               OpDecorate %w2 NoPerspective
               OpDecorate %w3 Location 3
               OpDecorate %w3 NoPerspective
               OpDecorate %near SpecId 0
               OpDecorate %far SpecId 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
         %PV = OpTypeStruct %v4float
     %out_PV = OpTypePointer Output %PV
    %out_v4f = OpTypePointer Output %v4float
  %out_float = OpTypePointer Output %float
     %in_v4f = OpTypePointer Input %v4float
     %uint_0 = OpConstant %uint 0
    %float_1 = OpConstant %float 1
       %near = OpSpecConstant %float 0
        %far = OpSpecConstant %float 1
        %pos = OpVariable %in_v4f Input
        %out = OpVariable %out_PV Output
         %w0 = OpVariable %out_float Output
         %w1 = OpVariable %out_float Output
         %w2 = OpVariable %out_float Output
         %w3 = OpVariable %out_float Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %p = OpLoad %v4float %pos
          %o = OpAccessChain %out_v4f %out %uint_0
               OpStore %o %p
          %z = OpCompositeExtract %float %p 2
          %w = OpCompositeExtract %float %p 3
      %range = OpFSub %float %far %near
         %zn = OpFDiv %float %z %w
        %rcp = OpFDiv %float %float_1 %w
         %zr = OpFMul %float %z %rcp
         %s0 = OpFMul %float %zn %range
         %v0 = OpFAdd %float %s0 %near
               OpStore %w0 %v0
         %v1 = OpExtInst %float %glsl Fma %zn %range %near
               OpStore %w1 %v1
         %v2 = OpExtInst %float %glsl Fma %zr %range %near
               OpStore %w2 %v2
         %s3 = OpFMul %float %zr %range
         %v3 = OpFAdd %float %s3 %near
               OpStore %w3 %v3
               OpReturn
               OpFunctionEnd
"#;

/// The per-sample depth candidates a multisampled fragment shader computes, each stored in its
/// own region of binding 0 after `FragCoord.z`'s (region `k + 1` for candidate `k`), compared like
/// a form with the stored depth of each sample (the computed per-sample depth):
/// - `derivative`: `z + dFdxFine(z) · (sx - 0.5) + dFdyFine(z) · (sy - 0.5)`, `z` the
///   `FragCoord.z` and `(sx, sy)` the `SamplePosition`;
/// - `window_div`, `window_div_fma`, `window_rcp_fma`, `window_rcp`: the vertex's window depth in
///   each of the vertex shader's four orders, without perspective, at the sample;
/// - each of those `_biased`: plus `FragCoord.z` less the same varying at the pixel's centre,
///   which carries the draw's depth bias.
const CANDIDATES: [&str; 9] = [
    "derivative",
    "window_div",
    "window_div_fma",
    "window_rcp_fma",
    "window_rcp",
    "window_div_biased",
    "window_div_fma_biased",
    "window_rcp_fma_biased",
    "window_rcp_biased",
];

/// The regions of binding 0 a fragment shader of `samples` writes: `FragCoord.z`'s, and at more
/// than one sample each candidate's.
fn regions(samples: u32) -> usize {
    if samples > 1 {
        1 + CANDIDATES.len()
    } else {
        1
    }
}

/// The fragment shader of `samples`: `FragCoord.z` into binding 0 at `(y · SIDE + x) · samples`,
/// plus the invocation's `SampleId` at more than one sample (which runs it per sample), and to
/// the color; at more than one sample also each of [`CANDIDATES`], `SIDE² · samples` words
/// further per candidate.
fn fragment(samples: u32) -> String {
    let per_sample = samples > 1;
    let words = SIDE * SIDE * samples;
    let (capability, interface, decorations, variables, candidates) = if per_sample {
        (
            "OpCapability SampleRateShading\n               OpCapability DerivativeControl\n               OpCapability InterpolationFunction",
            " %sid %spos %w0 %w1 %w2 %w3",
            "OpDecorate %sid BuiltIn SampleId\n               OpDecorate %sid Flat\n               \
             OpDecorate %spos BuiltIn SamplePosition\n               \
             OpDecorate %w0 Location 0\n               OpDecorate %w0 NoPerspective\n               \
             OpDecorate %w1 Location 1\n               OpDecorate %w1 NoPerspective\n               \
             OpDecorate %w2 Location 2\n               OpDecorate %w2 NoPerspective\n               \
             OpDecorate %w3 Location 3\n               OpDecorate %w3 NoPerspective",
            "%sid = OpVariable %in_int Input\n       %spos = OpVariable %in_v2f Input\n         \
             %w0 = OpVariable %in_float Input\n         %w1 = OpVariable %in_float Input\n         \
             %w2 = OpVariable %in_float Input\n         %w3 = OpVariable %in_float Input",
            format!(
                r#"
         %dx = OpDPdxFine %float %z
         %dy = OpDPdyFine %float %z
         %sp = OpLoad %v2float %spos
         %sx = OpCompositeExtract %float %sp 0
         %sy = OpCompositeExtract %float %sp 1
         %ox = OpFSub %float %sx %half
         %oy = OpFSub %float %sy %half
         %m1 = OpFMul %float %dx %ox
         %m2 = OpFMul %float %dy %oy
         %a1 = OpFAdd %float %z %m1
         %k0 = OpFAdd %float %a1 %m2
{varyings}{stores}"#,
                varyings = (0..4)
                    .map(|j| {
                        let (k, b) = (j + 1, j + 5);
                        format!(
                            "        %k{k} = OpExtInst %float %glsl InterpolateAtSample %w{j} %s\n        \
                             %c{j} = OpLoad %float %w{j}\n        %d{j} = OpFSub %float %z %c{j}\n        \
                             %k{b} = OpFAdd %float %k{k} %d{j}\n"
                        )
                    })
                    .collect::<String>(),
                stores = (0..CANDIDATES.len())
                    .map(|k| {
                        format!(
                            "       %ck{k} = OpIAdd %uint %i %uint_r{k}\n        \
                             %pk{k} = OpAccessChain %sb_float %buf %uint_0 %ck{k}\n               \
                             OpStore %pk{k} %k{k}\n"
                        )
                    })
                    .collect::<String>()
            ),
        )
    } else {
        ("", "", "", "", String::new())
    };
    let region_constants = if per_sample {
        (0..CANDIDATES.len())
            .map(|k| {
                format!(
                    "   %uint_r{k} = OpConstant %uint {}\n",
                    (k as u32 + 1) * words
                )
            })
            .collect::<String>()
    } else {
        String::new()
    };
    let index = if per_sample {
        "%s = OpLoad %int %sid\n         %us = OpBitcast %uint %s\n          %i = OpIAdd %uint %ps %us"
    } else {
        "%i = OpIAdd %uint %ps %uint_0"
    };
    format!(
        r#"
               OpCapability Shader
               {capability}
       %glsl = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %coord %color %buf{interface}
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %coord BuiltIn FragCoord
               OpDecorate %color Location 0
               {decorations}
               OpDecorate %arr ArrayStride 4
               OpMemberDecorate %Buf 0 Offset 0
               OpDecorate %Buf Block
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
    %v2float = OpTypeVector %float 2
    %v4float = OpTypeVector %float 4
        %arr = OpTypeRuntimeArray %float
        %Buf = OpTypeStruct %arr
     %in_v4f = OpTypePointer Input %v4float
     %in_v2f = OpTypePointer Input %v2float
   %in_float = OpTypePointer Input %float
    %out_v4f = OpTypePointer Output %v4float
     %in_int = OpTypePointer Input %int
     %sb_Buf = OpTypePointer StorageBuffer %Buf
   %sb_float = OpTypePointer StorageBuffer %float
     %uint_0 = OpConstant %uint 0
  %uint_side = OpConstant %uint {SIDE}
%uint_samples = OpConstant %uint {samples}
       %half = OpConstant %float 0.5
    %float_0 = OpConstant %float 0
      %zero2 = OpConstantComposite %v2float %float_0 %float_0
       %near = OpSpecConstant %float 0
        %far = OpSpecConstant %float 1
{region_constants}
      %coord = OpVariable %in_v4f Input
      %color = OpVariable %out_v4f Output
        {variables}
        %buf = OpVariable %sb_Buf StorageBuffer
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %c = OpLoad %v4float %coord
          %x = OpCompositeExtract %float %c 0
          %y = OpCompositeExtract %float %c 1
          %z = OpCompositeExtract %float %c 2
         %ux = OpConvertFToU %uint %x
         %uy = OpConvertFToU %uint %y
        %row = OpIMul %uint %uy %uint_side
         %px = OpIAdd %uint %row %ux
         %ps = OpIMul %uint %px %uint_samples
          {index}
          %p = OpAccessChain %sb_float %buf %uint_0 %i
               OpStore %p %z
{candidates}
          %o = OpCompositeConstruct %v4float %z %z %z %z
               OpStore %color %o
               OpReturn
               OpFunctionEnd
"#
    )
}

/// The readback of `samples`: each invocation `(x, y)` fetches every sample of the depth-aspect
/// view at binding 1 and writes its first channel into binding 0 at `(y · SIDE + x) · samples +
/// s`.
fn readback(samples: u32) -> String {
    let ms = u32::from(samples > 1);
    let mut constants = String::new();
    let mut body = String::new();
    for s in 0..samples {
        let (uint_s, int_s) = match s {
            0 => ("%uint_0".to_string(), "%int_0".to_string()),
            _ => {
                constants.push_str(&format!(
                    "        %uk{s} = OpConstant %uint {s}\n        %ik{s} = OpConstant %int {s}\n"
                ));
                (format!("%uk{s}"), format!("%ik{s}"))
            }
        };
        let operand = if samples > 1 {
            format!("Sample {int_s}")
        } else {
            "Lod %int_0".to_string()
        };
        body.push_str(&format!(
            "        %v{s} = OpImageFetch %v4float %t %xy {operand}\n\
             \x20       %d{s} = OpCompositeExtract %float %v{s} 0\n\
             \x20       %i{s} = OpIAdd %uint %ps {uint_s}\n\
             \x20       %p{s} = OpAccessChain %sb_float %buf %uint_0 %i{s}\n\
             \x20               OpStore %p{s} %d{s}\n"
        ));
    }
    format!(
        r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid %buf %tex
               OpExecutionMode %main LocalSize 8 8 1
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %arr ArrayStride 4
               OpMemberDecorate %Buf 0 Offset 0
               OpDecorate %Buf Block
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
               OpDecorate %tex DescriptorSet 0
               OpDecorate %tex Binding 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
     %v2int = OpTypeVector %int 2
    %v3uint = OpTypeVector %uint 3
    %v4float = OpTypeVector %float 4
        %img = OpTypeImage %float 2D 0 0 {ms} 1 Unknown
     %uc_img = OpTypePointer UniformConstant %img
        %arr = OpTypeRuntimeArray %float
        %Buf = OpTypeStruct %arr
     %sb_Buf = OpTypePointer StorageBuffer %Buf
   %sb_float = OpTypePointer StorageBuffer %float
  %in_v3uint = OpTypePointer Input %v3uint
     %uint_0 = OpConstant %uint 0
      %int_0 = OpConstant %int 0
  %uint_side = OpConstant %uint {SIDE}
%uint_samples = OpConstant %uint {samples}
{constants}
        %gid = OpVariable %in_v3uint Input
        %buf = OpVariable %sb_Buf StorageBuffer
        %tex = OpVariable %uc_img UniformConstant
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %g = OpLoad %v3uint %gid
         %gx = OpCompositeExtract %uint %g 0
         %gy = OpCompositeExtract %uint %g 1
         %ix = OpBitcast %int %gx
         %iy = OpBitcast %int %gy
         %xy = OpCompositeConstruct %v2int %ix %iy
          %t = OpLoad %img %tex
        %row = OpIMul %uint %gy %uint_side
         %px = OpIAdd %uint %row %gx
         %ps = OpIMul %uint %px %uint_samples
{body}
               OpReturn
               OpFunctionEnd
"#
    )
}

/// The objects one case creates, beyond [`Owned`]'s own kinds.
impl Owned<'_> {
    /// A device-local `info` image bound to memory of its own.
    pub(super) fn bound_image(
        &mut self,
        info: &vk::ImageCreateInfo<'_>,
    ) -> Result<vk::Image, String> {
        let image = self
            .image(info)
            .map_err(|error| format!("create an image: {}", result_name(error)))?;
        // SAFETY: the image was just created on this device.
        let requirements = unsafe { self.device().get_image_memory_requirements(image) };
        let memory_type = device_type(self.context, requirements.memory_type_bits)
            .ok_or("no memory type for an image")?;
        let memory = self.memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
        )?;
        // SAFETY: the image and memory are this device's, unbound, and the type fits.
        unsafe { self.device().bind_image_memory(image, memory, 0) }
            .map_err(|error| format!("bind an image: {}", result_name(error)))?;
        Ok(image)
    }

    pub(super) fn view(
        &mut self,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView, String> {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range(aspect));
        // SAFETY: the image is a live image of this device of `format`.
        let view = unsafe { self.device().create_image_view(&info, None) }
            .map_err(|error| format!("create an image view: {}", result_name(error)))?;
        self.objects.push(Object::ImageView(view));
        Ok(view)
    }

    pub(super) fn render_pass(
        &mut self,
        info: &vk::RenderPassCreateInfo<'_>,
    ) -> Result<vk::RenderPass, String> {
        // SAFETY: the device is live, and `info` is valid for the call.
        let pass = unsafe { self.device().create_render_pass(info, None) }
            .map_err(|error| format!("create a render pass: {}", result_name(error)))?;
        self.objects.push(Object::RenderPass(pass));
        Ok(pass)
    }

    pub(super) fn framebuffer(
        &mut self,
        pass: vk::RenderPass,
        views: &[vk::ImageView],
        side: u32,
    ) -> Result<vk::Framebuffer, String> {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(views)
            .width(side)
            .height(side)
            .layers(1);
        // SAFETY: the pass and views are live objects of this device, compatible with each other.
        let framebuffer = unsafe { self.device().create_framebuffer(&info, None) }
            .map_err(|error| format!("create a framebuffer: {}", result_name(error)))?;
        self.objects.push(Object::Framebuffer(framebuffer));
        Ok(framebuffer)
    }

    pub(super) fn module(&mut self, spv: &[u8], label: &str) -> Result<vk::ShaderModule, String> {
        let module = create_shader_module(self.context, spv, label)?;
        self.objects.push(Object::ShaderModule(module));
        Ok(module)
    }

    /// A pipeline layout of one set and no push constants.
    pub(super) fn plain_layout(
        &mut self,
        set: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout, String> {
        let sets = [set];
        let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&sets);
        // SAFETY: the set layout is a live object of this device.
        let layout = unsafe { self.device().create_pipeline_layout(&info, None) }
            .map_err(|error| format!("create a pipeline layout: {}", result_name(error)))?;
        self.objects.push(Object::PipelineLayout(layout));
        Ok(layout)
    }

    /// `count` descriptor sets of `layout` from a pool of their own of `sizes`.
    pub(super) fn sets(
        &mut self,
        layout: vk::DescriptorSetLayout,
        sizes: &[vk::DescriptorPoolSize],
        count: u32,
    ) -> Result<Vec<vk::DescriptorSet>, String> {
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(count)
            .pool_sizes(sizes);
        // SAFETY: the device is live, and `info` is valid for the call.
        let pool = unsafe { self.device().create_descriptor_pool(&info, None) }
            .map_err(|error| format!("create a descriptor pool: {}", result_name(error)))?;
        self.objects.push(Object::DescriptorPool(pool));
        let layouts = vec![layout; count as usize];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout are live objects of this device; the pool has room.
        unsafe { self.device().allocate_descriptor_sets(&allocate) }
            .map_err(|error| format!("allocate descriptor sets: {}", result_name(error)))
    }

    pub(super) fn graphics_pipeline(
        &mut self,
        info: &vk::GraphicsPipelineCreateInfo<'_>,
        label: &str,
    ) -> Result<vk::Pipeline, String> {
        // SAFETY: every object `info` names is a live object of this device.
        let pipelines = unsafe {
            self.device()
                .create_graphics_pipelines(vk::PipelineCache::null(), &[*info], None)
        }
        .map_err(|(_, error)| format!("create the {label} pipeline: {}", result_name(error)))?;
        self.objects.push(Object::Pipeline(pipelines[0]));
        Ok(pipelines[0])
    }
}

/// The first memory type among `bits` that is device-local, else the first among them.
fn device_type(context: &VulkanContext, bits: u32) -> Option<u32> {
    // SAFETY: the context's physical device belongs to its live instance.
    let properties = unsafe {
        context
            .instance
            .get_physical_device_memory_properties(context.physical)
    };
    let types = properties.memory_types_as_slice();
    let allowed = |index: &usize| bits & (1 << index) != 0;
    (0..types.len())
        .filter(allowed)
        .find(|&index| {
            types[index]
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .or_else(|| (0..types.len()).find(allowed))
        .map(|index| index as u32)
}

fn range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// A host-visible storage buffer of `words` words, each [`UNWRITTEN`]: the buffer and its mapping.
fn unwritten(owned: &mut Owned<'_>, words: usize) -> Result<(vk::Buffer, *mut u32), String> {
    let (buffer, mapped) =
        owned.host_buffer((words * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let mapped = mapped.cast::<u32>();
    // SAFETY: the mapping holds `words` words, host-coherent, and nothing on the device uses the
    // buffer yet.
    unsafe { std::slice::from_raw_parts_mut(mapped, words) }.fill(UNWRITTEN);
    Ok((buffer, mapped))
}

/// Run `case` at `samples` in every form: each form's report against the plain instance's stored
/// depth.
fn run_case(
    context: &VulkanContext,
    shaders: &Shaders,
    (format, samples, ordinary_clamp): (vk::Format, u32, bool),
    case: &Case,
) -> Result<(Vec<FormReport>, Vec<FormReport>), String> {
    let mut owned = Owned::new(context);
    let words = (SIDE * SIDE * samples) as usize;
    let sample_flags = vk::SampleCountFlags::from_raw(samples);
    let depth_aspects = match format {
        vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::DEPTH,
    };
    let image_info = |format: vk::Format, usage: vk::ImageUsageFlags| {
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: SIDE,
                height: SIDE,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(sample_flags)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    };
    // The vertices.
    let bytes: Vec<u8> = case
        .vertices
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let (vertices, mapped) =
        owned.host_buffer(bytes.len() as u64, vk::BufferUsageFlags::VERTEX_BUFFER)?;
    // SAFETY: the mapping holds the buffer's bytes, and nothing on the device uses it yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped, bytes.len()) };
    // Per form: its storage buffer, color, depth (where it has one) and pass.
    let mut buffers = Vec::new();
    let mut targets = Vec::new();
    let mut stored_view = None;
    for form in FORMS {
        buffers.push(unwritten(&mut owned, words * regions(samples))?);
        let color = owned.bound_image(&image_info(
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ))?;
        let mut views =
            vec![owned.view(color, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR)?];
        if form.depth() {
            let usage =
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
            let depth = owned.bound_image(&image_info(format, usage))?;
            views.push(owned.view(depth, format, depth_aspects)?);
            if form == Form::Plain {
                stored_view = Some(owned.view(depth, format, vk::ImageAspectFlags::DEPTH)?);
            }
        }
        let pass = render_pass(&mut owned, format, sample_flags, form)?;
        let framebuffer = owned.framebuffer(pass, &views, SIDE)?;
        targets.push((pass, framebuffer));
    }
    let stored_view = stored_view.expect("the plain form holds the depth");
    let (stored, stored_words) = unwritten(&mut owned, words)?;
    // The fragment stage's set: one storage buffer per form.
    let fragment_binding = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let fragment_set_layout = owned.descriptor_set_layout(&fragment_binding)?;
    let fragment_layout = owned.plain_layout(fragment_set_layout)?;
    let fragment_sets = owned.sets(
        fragment_set_layout,
        &[vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(FORMS.len() as u32)],
        FORMS.len() as u32,
    )?;
    let (readback_layout, readback_set) = readback_set_layout(&mut owned)?;
    let buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
        .iter()
        .map(|&(buffer, _)| {
            [vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(vk::WHOLE_SIZE)]
        })
        .collect();
    let stored_info = [vk::DescriptorBufferInfo::default()
        .buffer(stored)
        .range(vk::WHOLE_SIZE)];
    let image_info_read = [vk::DescriptorImageInfo::default()
        .image_view(stored_view)
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)];
    let mut writes: Vec<vk::WriteDescriptorSet<'_>> = fragment_sets
        .iter()
        .zip(&buffer_infos)
        .map(|(&set, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect();
    writes.push(
        vk::WriteDescriptorSet::default()
            .dst_set(readback_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&stored_info),
    );
    writes.push(
        vk::WriteDescriptorSet::default()
            .dst_set(readback_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&image_info_read),
    );
    // SAFETY: the sets, buffers and view are live objects of this device; nothing uses the sets.
    unsafe { owned.device().update_descriptor_sets(&writes, &[]) };
    // The pipelines.
    let vertex = owned.module(&shaders.vertex, "depth source vertex")?;
    let fragment = owned.module(
        Shaders::of(&shaders.fragment, samples),
        "depth source fragment",
    )?;
    let readback = owned.compute_pipeline(
        Shaders::of(&shaders.readback, samples),
        "depth source readback",
        readback_layout,
    )?;
    let mut pipelines = Vec::new();
    for (form, &(pass, _)) in FORMS.iter().zip(&targets) {
        pipelines.push(pipeline(
            &mut owned,
            (vertex, fragment),
            fragment_layout,
            pass,
            sample_flags,
            (*form, ordinary_clamp),
            case,
        )?);
    }
    owned.submit("the depth source probe's draws", |device, command| {
        let clears = [
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: SIDE,
                height: SIDE,
            },
        };
        for (index, form) in FORMS.iter().enumerate() {
            let (pass, framebuffer) = targets[index];
            let count = if form.depth() { 2 } else { 1 };
            let begin = vk::RenderPassBeginInfo::default()
                .render_pass(pass)
                .framebuffer(framebuffer)
                .render_area(area)
                .clear_values(&clears[..count]);
            // SAFETY: the command buffer is recording; every object it names is live.
            unsafe {
                device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines[index],
                );
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    fragment_layout,
                    0,
                    &[fragment_sets[index]],
                    &[],
                );
                device.cmd_bind_vertex_buffers(command, 0, &[vertices], &[0]);
                device.cmd_draw(command, case.vertices.len() as u32, 1, 0, 0);
                device.cmd_end_render_pass(command);
            }
        }
        let to_host = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        // SAFETY: as above. The plain pass's dependency makes its depth writes and layout
        // transition visible to the readback.
        unsafe {
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, readback);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                readback_layout,
                0,
                &[readback_set],
                &[],
            );
            device.cmd_dispatch(command, SIDE / 8, SIDE / 8, 1);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &to_host,
                &[],
                &[],
            );
        }
    })?;
    let read_region = |mapped: *mut u32, region: usize| -> Vec<u32> {
        // SAFETY: the mapping holds `words` host-coherent words per region the shader writes,
        // and the submission that wrote them completed, its writes made available to the host by
        // the barrier.
        unsafe { std::slice::from_raw_parts(mapped.add(region * words), words) }.to_vec()
    };
    let read = |mapped: *mut u32| read_region(mapped, 0);
    let stored = read(stored_words);
    let reference = read(buffers[0].1);
    // Metal's clamp: to the range in clamp mode, to `[0, 1]` in clip mode.
    let (low, high) = match case.clamp {
        true => (
            case.range[0].min(case.range[1]),
            case.range[0].max(case.range[1]),
        ),
        false => (0.0, 1.0),
    };
    let reports = FORMS
        .iter()
        .zip(&buffers)
        .map(|(form, &(_, mapped))| {
            compare(form.name(), &read(mapped), &reference, &stored, low, high)
        })
        .collect();
    let candidates = match samples {
        1 => Vec::new(),
        _ => CANDIDATES
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let seen = read_region(buffers[0].1, k + 1);
                compare(name, &seen, &reference, &stored, low, high)
            })
            .collect(),
    };
    Ok((reports, candidates))
}

/// The readback's pipeline layout and its one set: the stored depth's storage buffer (binding 0)
/// and its depth-aspect view (binding 1).
fn readback_set_layout(
    owned: &mut Owned<'_>,
) -> Result<(vk::PipelineLayout, vk::DescriptorSet), String> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let set_layout = owned.descriptor_set_layout(&bindings)?;
    let layout = owned.plain_layout(set_layout)?;
    let set = owned.sets(
        set_layout,
        &[
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1),
        ],
        1,
    )?[0];
    Ok((layout, set))
}

/// One form's report: `seen` (its `FragCoord.z` bits per sample) against `stored` over the
/// samples `reference` (the plain instance's) covered.
fn compare(
    form: &'static str,
    seen: &[u32],
    reference: &[u32],
    stored: &[u32],
    low: f32,
    high: f32,
) -> FormReport {
    let mut report = FormReport {
        form,
        covered: 0,
        same_coverage: true,
        equal_clamped: 0,
        equal_raw: 0,
        max_ulp: 0,
        first_differences: Vec::new(),
    };
    for (index, ((&z, &r), &d)) in seen.iter().zip(reference).zip(stored).enumerate() {
        if (z == UNWRITTEN) != (r == UNWRITTEN) {
            report.same_coverage = false;
        }
        if r == UNWRITTEN || z == UNWRITTEN {
            continue;
        }
        report.covered += 1;
        let clamped = f32::from_bits(z).clamp(low, high).to_bits();
        if z == d {
            report.equal_raw += 1;
        }
        if clamped == d {
            report.equal_clamped += 1;
            continue;
        }
        let ulp = (i64::from(clamped as i32) - i64::from(d as i32)).unsigned_abs();
        report.max_ulp = report.max_ulp.max(ulp);
        if report.first_differences.len() < 4 {
            report.first_differences.push([index as u32, z, d]);
        }
    }
    report
}

/// The render pass of `form`: the color, and the depth attachment where the form holds it, the
/// plain form's left for the readback in `DEPTH_STENCIL_READ_ONLY_OPTIMAL` with its writes made
/// visible to compute.
fn render_pass(
    owned: &mut Owned<'_>,
    format: vk::Format,
    samples: vk::SampleCountFlags,
    form: Form,
) -> Result<vk::RenderPass, String> {
    // The tested draw's color is copied to a buffer after its pass; its depth is the plain
    // draw's, loaded.
    let (color_store, color_last) = match form {
        Form::Tested => (
            vk::AttachmentStoreOp::STORE,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ),
        _ => (
            vk::AttachmentStoreOp::DONT_CARE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
    };
    let mut attachments = vec![vk::AttachmentDescription::default()
        .format(vk::Format::R32_SFLOAT)
        .samples(samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(color_store)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(color_last)];
    if form.depth() {
        let (store, last) = match form {
            Form::Plain => (
                vk::AttachmentStoreOp::STORE,
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ),
            _ => (
                vk::AttachmentStoreOp::DONT_CARE,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ),
        };
        let (load, stencil_load, first) = match form {
            Form::Tested => (
                vk::AttachmentLoadOp::LOAD,
                vk::AttachmentLoadOp::DONT_CARE,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ),
            _ => (
                vk::AttachmentLoadOp::CLEAR,
                vk::AttachmentLoadOp::CLEAR,
                vk::ImageLayout::UNDEFINED,
            ),
        };
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(samples)
                .load_op(load)
                .store_op(store)
                .stencil_load_op(stencil_load)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(first)
                .final_layout(last),
        );
    }
    let colors = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors);
    if form.depth() {
        subpass = subpass.depth_stencil_attachment(&depth);
    }
    let subpasses = [subpass];
    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_WRITE,
        )
        .dst_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)];
    let copied = [vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
    let mut info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    match form {
        Form::Plain => info = info.dependencies(&dependencies),
        Form::Tested => info = info.dependencies(&copied),
        _ => {}
    }
    owned.render_pass(&info)
}

/// The pipeline of `form` for `case`: the viewport's range (y flipped, as the product draws),
/// depth clamp, the product's depth bias, and the form's depth state; in clip mode under
/// `ordinary_clamp`, depth clamp on with the user-defined range `[0, 1]` and clipping kept.
fn pipeline(
    owned: &mut Owned<'_>,
    (vertex, fragment): (vk::ShaderModule, vk::ShaderModule),
    layout: vk::PipelineLayout,
    pass: vk::RenderPass,
    samples: vk::SampleCountFlags,
    (form, ordinary_clamp): (Form, bool),
    case: &Case,
) -> Result<vk::Pipeline, String> {
    let profile = owned
        .context
        .opened
        .as_ref()
        .map_err(|error| format!("the context's profile: {error}"))?;
    let [constant, slope, clamp] = case.bias;
    let bias = metal2vulkan::metal::state::depth_bias(
        metal2vulkan::vk::PrimitiveTopology::TRIANGLE_LIST,
        constant,
        slope,
        clamp,
        Some(metal2vulkan::metal::PixelFormat::Depth32Float),
        profile,
    )
    .map_err(|error| format!("the product's depth bias: {error}"))?;
    if bias.unit != metal2vulkan::metal::state::DepthBiasUnit::Device {
        return Err(format!(
            "a depth bias in {:?}, not the device's unit",
            bias.unit
        ));
    }
    // The viewport's depth range, near and far, as specialization constants 0 and 1.
    let range_entries = [
        vk::SpecializationMapEntry {
            constant_id: 0,
            offset: 0,
            size: 4,
        },
        vk::SpecializationMapEntry {
            constant_id: 1,
            offset: 4,
            size: 4,
        },
    ];
    let range_bytes: Vec<u8> = case.range.iter().flat_map(|v| v.to_le_bytes()).collect();
    let range_info = vk::SpecializationInfo::default()
        .map_entries(&range_entries)
        .data(&range_bytes);
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main")
            .specialization_info(&range_info),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main")
            .specialization_info(&range_info),
    ];
    let bindings = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: 16,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let attributes = [vk::VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: 0,
    }];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let side = SIDE as f32;
    let viewports = [vk::Viewport {
        x: 0.0,
        y: side,
        width: side,
        height: -side,
        min_depth: case.range[0],
        max_depth: case.range[1],
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: SIDE,
            height: SIDE,
        },
    }];
    let clamp01 = ordinary_clamp && !case.clamp;
    let range01 = [0.0f32, 1.0];
    let mut control = DepthClampControlCreateInfo {
        s_type: <DepthClampControlCreateInfo as vk::TaggedStructure>::STRUCTURE_TYPE,
        p_next: std::ptr::null(),
        depth_clamp_mode: 1,
        depth_clamp_range: &range01,
    };
    let mut depth_clip =
        vk::PipelineRasterizationDepthClipStateCreateInfoEXT::default().depth_clip_enable(true);
    let mut viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    if clamp01 {
        viewport = viewport.push_next(&mut control);
    }
    let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(case.clamp || clamp01)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(bias.enable)
        .depth_bias_constant_factor(bias.constant_factor)
        .depth_bias_clamp(bias.clamp)
        .depth_bias_slope_factor(bias.slope_factor)
        .line_width(1.0);
    if clamp01 {
        raster = raster.push_next(&mut depth_clip);
    }
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples);
    let depth_state = match form {
        Form::Plain => vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::ALWAYS),
        Form::KeptAlways => vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_compare_op(vk::CompareOp::ALWAYS),
        Form::Tested => vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_compare_op(vk::CompareOp::EQUAL),
        Form::Out | Form::KeptOff => vk::PipelineDepthStencilStateCreateInfo::default(),
    };
    let blend = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::R)];
    let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blending)
        .layout(layout)
        .render_pass(pass)
        .subpass(0);
    if form.depth() {
        info = info.depth_stencil_state(&depth_state);
    }
    owned.graphics_pipeline(&info, &format!("depth source {}", form.name()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The step's shaders, at both sample counts, assemble and pass `spirv-val` for Vulkan 1.2.
    #[test]
    fn the_shaders_assemble_and_validate() {
        let shaders = Shaders::assemble(true).unwrap_or_else(|error| panic!("{error}"));
        assert!(!shaders.vertex.is_empty());
        for samples in [1, MSAA] {
            assert!(!Shaders::of(&shaders.fragment, samples).is_empty());
            assert!(!Shaders::of(&shaders.readback, samples).is_empty());
        }
    }

    /// A form's report counts only the samples the plain instance covered, compares the clamped
    /// `FragCoord.z` and the raw one, and flags a coverage that differs.
    #[test]
    fn a_report_compares_the_clamped_depth_over_the_covered_samples() {
        let bits = f32::to_bits;
        let reference = [bits(0.5), bits(1.25), UNWRITTEN, bits(0.25)];
        let stored = [bits(0.5), bits(1.0), UNWRITTEN, bits(0.25)];
        let seen = [bits(0.5), bits(1.25), UNWRITTEN, bits(0.25)];
        let report = compare("out", &seen, &reference, &stored, 0.0, 1.0);
        assert_eq!(
            (report.covered, report.equal_clamped, report.equal_raw),
            (3, 3, 2)
        );
        assert!(report.same_coverage);
        assert_eq!(report.max_ulp, 0);
        let off = [
            bits(0.5),
            bits(1.25),
            bits(0.75),
            f32::from_bits(bits(0.25) + 2).to_bits(),
        ];
        let report = compare("kept_off", &off, &reference, &stored, 0.0, 1.0);
        assert!(!report.same_coverage);
        assert_eq!((report.covered, report.equal_clamped), (3, 2));
        assert_eq!(report.max_ulp, 2);
        assert_eq!(report.first_differences, [[3, bits(0.25) + 2, bits(0.25)]]);
    }
}
