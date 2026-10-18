//! The extended-range blend table (`metal2vulkan_devtools::xr_blends`) on the device the cfg
//! selects, under the validation layer, against Metal's bytes (plain Metal on an
//! Apple M4 under macOS 27.0.1).
//!
//! For each format and output type (`half4` through the owned fixture `rt6_samples`'s
//! `rt6_fill_rgba16float`, `float4` through its `rt6_fill_rgba32float`), every configuration of
//! the table makes its pipeline: one the table says metal2vulkan makes must be made, one it says
//! is refused must be refused with that `ExtendedRangeBlend` gap, on a device with raster-order
//! attachment access (lavapipe) too, which blends over the attachment's input attachment; any other
//! refusal fails. Passes that load and store the target then draw each made configuration into its
//! own band under its scissor and blend color, a few configurations to a pass ([`PASS_BANDS`]),
//! every band holding the table's texels and outputs, and each band's bytes must be Metal's: plane
//! 0 through a blit, a two-plane format's alpha bytes as `round(255 a)` of a `float` read. Where it
//! lies: a texture of the device's own, a placement heap's, and (the two-plane formats) one over
//! the planes of a `'b3a8'` surface, whose bytes are read where Metal keeps them. A pass that
//! announces `extended-range-bits-kept` (a heap's two-plane texture, a shadow of the heap's bytes
//! the load pass does not take) keeps the bits the codes leave unused in the texels its draws do
//! not write, where Metal's pass rewrites them: its bands are compared with those bits as Metal's
//! pass leaves them ([`as_metal_leaves`]), the codes exactly. A differing band is printed on an
//! `XR-TABLE-DIFFERS` line.
//!
//! The table of the probe built without fast math (`xr_blends::safe_table`) is
//! drawn the same way through the fixture built so, its `air.compile.fast_math_enable` switched to
//! `_disable` ([`Math::Safe`]).
//!
//! The second source's table without fast math (`xr_blends::dual_table`) is drawn
//! through the owned fixture `xrb_dual`, built without fast math, whose `xrb_dual_half` and
//! `xrb_dual_float` return a second output (index 1) read from buffer 2, which holds the table's
//! second outputs ([`Math::Second`]).
//!
//! The table of a computed `float4` output under fast math (`xr_blends::computed_table`) is drawn
//! through the owned fixture `xrb_computed`, whose `xrb_computed_float` returns its value times
//! buffer 2's ones ([`Math::Computed`]): a row the table says is made must announce no
//! `extended-range-float-blend-source` (the `float` output kind; a `half4` output's is
//! `extended-range-blend-source`) and store Metal's bytes; one it says
//! is `announced` must announce it, and its band is not compared. The lane asks for both kinds at
//! every occurrence, which announces them at each draw's bind, not at the making: an announced row
//! draws in a pass of its own.

use super::degrade_notices::counts;
use super::extended_range_targets::{dispatch, kernel, reads};
use super::resolves::{lane, lane_with, quiet, Lane};
use super::{buffer_with, read_buffer, WAIT};
use crate::candidate::runtime_samples::{library, shape};
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::xr_blends::{self, Layout, Table, BAND, BAND_HEIGHT, WIDTH};
use metal2vulkan_runtime::{
    BlitOption, Buffer, Degradation, Error, HeapDescriptor, HeapType, IoSurfacePlane,
    IoSurfaceYCbCr, Library, LoadAction, PipelineOptions, Region, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, StoreAction, Texture,
    TextureDescriptor,
};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Where a target lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lies {
    /// A texture of the device's own.
    Own,
    /// A placement heap's texture at offset 0.
    Heap,
    /// A texture over plane 0 of a two-plane `'b3a8'` surface, plane 1 right after it.
    Surface,
}

/// The math the fixture is built under, and the table of the probe built so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Math {
    /// Fast math: `xr_blends::table`, the fixture as committed.
    Fast,
    /// Without fast math: `xr_blends::safe_table`, the fixture's `air.compile.fast_math_enable`
    /// switched to `_disable`.
    Safe,
    /// Without fast math, the second source's: `xr_blends::dual_table` through the owned fixture
    /// `xrb_dual` (built without fast math).
    Second,
    /// Fast math, a `float4` output computed: `xr_blends::computed_table` through the owned
    /// fixture `xrb_computed`.
    Computed,
    /// [`Math::Second`]'s function drawn with blending disabled, its second output read by no
    /// blend: `xr_blends::dual_unblended_table`.
    SecondUnblended,
}

impl Math {
    /// The fixture's vertex function and its fragment function of a `half4` or `float4` output.
    fn functions(self, half: bool) -> (&'static str, &'static str) {
        match (self, half) {
            (Math::Second | Math::SecondUnblended, true) => ("xrb_dual_vertex", "xrb_dual_half"),
            (Math::Second | Math::SecondUnblended, false) => ("xrb_dual_vertex", "xrb_dual_float"),
            (Math::Computed, _) => ("xrb_computed_vertex", "xrb_computed_float"),
            (_, true) => ("rt6_fill_vertex", "rt6_fill_rgba16float"),
            (_, false) => ("rt6_fill_vertex", "rt6_fill_rgba32float"),
        }
    }
}

/// The fixture `rt6_samples`'s library on `lane`'s device built under `math`; under
/// [`Math::Second`] the fixture `xrb_dual`'s, and under [`Math::Computed`] `xrb_computed`'s.
fn fixture_library(lane: &Lane, math: Math) -> Library {
    use crate::air_library::AirLibrary;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble_text};
    match math {
        Math::Fast => library(&lane.device).unwrap(),
        Math::Safe => {
            let text = std::fs::read_to_string(engine_dir().join("rt6_samples.ll")).unwrap();
            let enable = "air.compile.fast_math_enable";
            assert_eq!(text.matches(enable).count(), 1);
            let text = text.replace(enable, "air.compile.fast_math_disable");
            let bitcode = llvm_assemble_text(&text).unwrap();
            lane.device.new_library_with_air(&bitcode).unwrap()
        }
        Math::Second | Math::SecondUnblended => {
            let text = std::fs::read_to_string(engine_dir().join("xrb_dual.ll")).unwrap();
            assert_eq!(text.matches("air.compile.fast_math_disable").count(), 1);
            let bitcode = llvm_assemble_text(&text).unwrap();
            lane.device.new_library_with_air(&bitcode).unwrap()
        }
        Math::Computed => {
            let text = std::fs::read_to_string(engine_dir().join("xrb_computed.ll")).unwrap();
            assert_eq!(text.matches("air.compile.fast_math_enable").count(), 1);
            let bitcode = llvm_assemble_text(&text).unwrap();
            lane.device.new_library_with_air(&bitcode).unwrap()
        }
    }
}

/// `'b3a8'`.
const B3A8: u32 = u32::from_be_bytes(*b"b3a8");

/// A configuration's pipeline into `pf` from the fixture's (`library`, built under `math`)
/// `half4` or `float4` fill.
fn pipeline(
    (lane, library, math): (&Lane, &Library, Math),
    pf: PixelFormat,
    half: bool,
    c: &xr_blends::Config,
) -> Result<RenderPipelineState, Error> {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let (vertex, fragment) = math.functions(half);
    let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
    descriptor.fragment = Some(function(fragment));
    let mut color = attachment(pf, c);
    color.blending_enabled = math != Math::SecondUnblended;
    descriptor.color_attachments[0] = Some(color);
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A color attachment of `pf` under configuration `c`'s blend and write mask.
fn attachment(pf: PixelFormat, c: &xr_blends::Config) -> ColorAttachmentDescriptor {
    let factor = |k: usize| BlendFactor::try_from(c.equation[k] as u32).unwrap();
    let op = |k: usize| BlendOperation::try_from(c.equation[k] as u32).unwrap();
    let mut color = ColorAttachmentDescriptor::new(pf);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = factor(0);
    color.destination_rgb_blend_factor = factor(1);
    color.rgb_blend_operation = op(2);
    color.source_alpha_blend_factor = factor(3);
    color.destination_alpha_blend_factor = factor(4);
    color.alpha_blend_operation = op(5);
    color.write_mask = ColorWriteMask::try_from(c.mask as u32).unwrap();
    color
}

/// The target's height: a band per configuration.
fn height(table: &Table) -> u64 {
    (BAND_HEIGHT * table.configs.len()) as u64
}

/// A target of `pf` lying as `lies`, of every band of the table holding the band's texels
/// `(plane, alpha)`, and the surface it lies over.
fn target(
    lane: &Lane,
    pf: PixelFormat,
    lies: Lies,
    (plane, alpha): (&[u8], &[u8]),
    table: &Table,
) -> (Texture, Option<Buffer>) {
    let (w, h) = (WIDTH as u64, height(table));
    let bands = table.configs.len();
    let plane: Vec<u8> = plane.repeat(bands);
    let alpha: Vec<u8> = alpha.repeat(bands);
    let mut descriptor = TextureDescriptor::texture_2d(pf, w, h, false);
    descriptor.usage =
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    let row = plane.len() as u64 / h;
    if lies == Lies::Surface {
        // Plane 0's rows, then plane 1's alpha bytes right after them.
        let mut bytes = plane;
        let alpha_offset = bytes.len() as u64;
        bytes.extend(&alpha);
        let surface = buffer_with(&lane.device, &bytes).unwrap();
        let facts = IoSurfaceYCbCr::new(B3A8, 0, [w, h], alpha_offset, w);
        let t = surface
            .new_texture_with_iosurface_plane(
                &descriptor,
                IoSurfacePlane::new(0, row, true).with_ycbcr(facts),
            )
            .unwrap_or_else(|e| panic!("{pf:?} over a surface: {e}"));
        return (t, Some(surface));
    }
    let t = match lies {
        Lies::Heap => {
            let answer = lane
                .device
                .heap_texture_size_and_align(&descriptor)
                .unwrap();
            let heap = lane
                .device
                .new_heap(&HeapDescriptor::new(
                    HeapType::Placement,
                    answer.size,
                    ResourceOptions::shared(),
                ))
                .unwrap();
            heap.new_texture_at_offset(&descriptor, 0)
        }
        _ => lane.device.new_texture(&descriptor),
    }
    .unwrap_or_else(|e| panic!("{pf:?} {lies:?}: {e}"));
    if !alpha.is_empty() {
        let values: Vec<u8> = alpha
            .iter()
            .flat_map(|&b| [0.0, 0.0, 0.0, f32::from(b) / 255.0])
            .flat_map(f32::to_le_bytes)
            .collect();
        let values = buffer_with(&lane.device, &values).unwrap();
        dispatch(
            lane,
            &kernel(lane, "xr2p_write_float"),
            &t,
            (&values, 0),
            (w, h),
        );
    }
    t.replace_region(Region::new_2d(0, 0, w, h), 0, 0, &plane, row, 0)
        .unwrap();
    (t, None)
}

/// The target's plane-0 bytes and a two-plane format's alpha bytes after the pass.
fn bytes_of(
    lane: &Lane,
    (t, surface): (&Texture, Option<&Buffer>),
    layout: Layout,
    table: &Table,
) -> (Vec<u8>, Vec<u8>) {
    let (w, h) = (WIDTH as u64, height(table));
    let plane_length = layout.bytes() as u64 * w * h;
    if let Some(surface) = surface {
        let bytes = read_buffer(surface, 0, plane_length + w * h).unwrap();
        let (plane, alpha) = bytes.split_at(plane_length as usize);
        return (plane.to_vec(), alpha.to_vec());
    }
    let plane = lane.read(t, (0, 0), layout.bytes() as u64, BlitOption::None);
    if layout != Layout::Planes {
        return (plane, Vec::new());
    }
    let alpha = reads(lane, &kernel(lane, "xr2p_read_float"), t, (w, h))
        .chunks(16)
        .map(|v| {
            let a = f32::from_le_bytes([v[12], v[13], v[14], v[15]]);
            (a * 255.0).round_ties_even() as u8
        })
        .collect();
    (plane, alpha)
}

/// Each made configuration's draw into its band, under its pipeline and blend color.
fn blend_bands(
    e: &mut RenderCommandEncoder,
    made: &[(&xr_blends::Row, RenderPipelineState)],
    table: &Table,
) {
    for (row, p) in made {
        let c = &table.configs[row.config];
        e.set_render_pipeline_state(p).unwrap();
        let y = (row.config * BAND_HEIGHT) as u64;
        e.set_scissor_rect(ScissorRect::new(0, y, WIDTH as u64, BAND_HEIGHT as u64))
            .unwrap();
        let [r, g, b, a] = c.color;
        e.set_blend_color(r, g, b, a);
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
}

/// The made configurations one pass draws. Lavapipe compiles a pipeline's fragment shader at its
/// first draw, a blend over the input attachment without fast math in seconds, so a pass of every
/// configuration would outlast the lane's wait ([`WAIT`]); each pass loads what the one before
/// stored.
const PASS_BANDS: usize = 4;

/// What one case found: pipelines checked, bands drawn, those whose unused bits the pass
/// announced it keeps, the differences, and the longest pass of made bands, from its encoding
/// until it completed.
#[derive(Default)]
struct Found {
    checked: usize,
    drawn: usize,
    bits_kept: usize,
    wrong: Vec<String>,
    slowest: Duration,
}

/// A band's plane-0 bytes of `layout` with the bits its codes leave unused as Metal's pass leaves
/// them in each tile a draw reaches: a two-plane word's top two bits zero, a `BGR10_XR` word's
/// ones, and a `BGRA10_XR` field's six low bits zero, its alpha code clamped to `[384, 894]`.
fn as_metal_leaves(layout: Layout, band: &[u8]) -> Vec<u8> {
    match layout {
        Layout::Planes | Layout::Bgr10 => band
            .chunks(4)
            .flat_map(|w| {
                let w = u32::from_le_bytes([w[0], w[1], w[2], w[3]]) & 0x3FFF_FFFF;
                let top = if layout == Layout::Bgr10 {
                    0xC000_0000
                } else {
                    0
                };
                (w | top).to_le_bytes()
            })
            .collect(),
        Layout::Bgra10 => band
            .chunks(2)
            .enumerate()
            .flat_map(|(i, f)| {
                let code = u16::from_le_bytes([f[0], f[1]]) >> 6;
                let code = if i % 4 == 3 {
                    code.clamp(384, 894)
                } else {
                    code
                };
                (code << 6).to_le_bytes()
            })
            .collect(),
    }
}

/// Each format of `formats`, from `half4` and `float4` outputs of the fixture built under
/// `math`, into targets lying as each of `lies` (a surface for the two-plane formats alone):
/// every configuration's outcome is the table's, and every made one's band Metal's bytes.
fn check(formats: &[u32], lies: &[Lies], math: Math) {
    let kept = Degradation::ExtendedRangeBitsKept;
    let sources = [
        Degradation::ExtendedRangeBlendSource,
        Degradation::ExtendedRangeFloatBlendSource,
    ];
    let (messages, _cache, lane) =
        lane_with(|options| options.announce_each = [kept].into_iter().chain(sources).collect());
    let name = lane.device.name().to_owned();
    let table = match math {
        Math::Fast => xr_blends::table(),
        Math::Safe => xr_blends::safe_table(),
        Math::Second => xr_blends::dual_table(),
        Math::SecondUnblended => xr_blends::dual_unblended_table(),
        Math::Computed => xr_blends::computed_table(),
    };
    let mut found = Found::default();
    let fixture = fixture_library(&lane, math);
    for &raw in formats {
        let pf = PixelFormat::try_from(raw).unwrap();
        let layout = Layout::of(raw).unwrap();
        let (plane, alpha) = xr_blends::texels(raw);
        for half in [true, false] {
            let kind = if half { "h" } else { "f" };
            let rows = table
                .rows
                .iter()
                .filter(|r| r.format == raw && r.half == half);
            let mut made = Vec::new();
            // The made pipelines the table says announce their output's blend-source kind, which
            // a kind announced at every occurrence announces at each draw's bind, not at the
            // making.
            let mut announced = Vec::new();
            let (source, other) = match half {
                true => (sources[0], sources[1]),
                false => (sources[1], sources[0]),
            };
            for row in rows {
                let c = &table.configs[row.config];
                let what = format!("{raw}.{kind} cfg {}", row.config);
                found.checked += 1;
                let made_now = pipeline((&lane, &fixture, math), pf, half, c);
                match made_now {
                    Ok(p) if row.ours == "made" => made.push((row, p)),
                    Ok(p) if row.ours == "announced" => announced.push((row, p)),
                    Ok(_) => found
                        .wrong
                        .push(format!("{what}: made, the table says {}", row.ours)),
                    Err(Error::Unsupported(u)) => match u.feature {
                        Feature::ExtendedRangeBlend(g) if u.needs == Needs::NotImplemented => {
                            let gap = format!("{g:?}");
                            if gap != row.ours {
                                found
                                    .wrong
                                    .push(format!("{what}: refused {gap}, the table {}", row.ours));
                            }
                        }
                        _ => found.wrong.push(format!("{what}: refused {u:?}")),
                    },
                    Err(e) => found.wrong.push(format!("{what}: {e}")),
                }
            }
            if made.is_empty() && announced.is_empty() {
                continue;
            }
            // The outputs of every band, the same in each.
            let values = buffer_with(
                &lane.device,
                &xr_blends::sources(half).repeat(table.configs.len()),
            )
            .unwrap();
            // `xrb_dual` reads the shape's first word, the width.
            let fill_shape = shape(&lane.device, WIDTH as u32, 1).unwrap();
            // The second outputs of every band, the same in each.
            let seconds = buffer_with(
                &lane.device,
                &xr_blends::second_sources(half).repeat(table.configs.len()),
            )
            .unwrap();
            // The computed output's ones, a `float4` per texel of every band.
            let ones = buffer_with(
                &lane.device,
                &1.0f32.to_le_bytes().repeat(4 * BAND * table.configs.len()),
            )
            .unwrap();
            for &where_ in lies {
                if where_ == Lies::Surface && layout != Layout::Planes {
                    continue;
                }
                let (t, surface) = target(&lane, pf, where_, (&plane, &alpha), &table);
                let mut pass = RenderPassDescriptor::new();
                let attachment = &mut pass.color_attachments[0];
                attachment.texture = Some(t.clone());
                attachment.load_action = LoadAction::Load;
                attachment.store_action = StoreAction::Store;
                let draw = |bands: &[(&xr_blends::Row, RenderPipelineState)], what: &str| {
                    lane.run(&pass, |_, e, _| {
                        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
                        e.set_fragment_buffer(Some(&fill_shape), 0, 1).unwrap();
                        match math {
                            Math::Second | Math::SecondUnblended => {
                                e.set_fragment_buffer(Some(&seconds), 0, 2).unwrap()
                            }
                            Math::Computed => e.set_fragment_buffer(Some(&ones), 0, 2).unwrap(),
                            Math::Fast | Math::Safe => {}
                        }
                        blend_bands(e, bands, &table);
                    })
                    .unwrap_or_else(|e| panic!("{raw}.{kind} {where_:?}: {what}: {e}"));
                    counts(&messages.take_notices())
                };
                let mut notices = BTreeMap::new();
                for bands in made.chunks(PASS_BANDS) {
                    let start = Instant::now();
                    for (degradation, n) in draw(bands, "a pass") {
                        *notices.entry(degradation).or_insert(0) += n;
                    }
                    found.slowest = found.slowest.max(start.elapsed());
                }
                let bits_kept = notices.contains_key(&kept);
                // The made bands announce no blend-source kind; each announced band its own
                // output's kind, never the other's (each in a pass of its own, after the made
                // bands' bytes are read; its band is not compared).
                if notices.contains_key(&source) || notices.contains_key(&other) {
                    found.wrong.push(format!(
                        "{raw}.{kind} {where_:?}: the made bands announced {notices:?}"
                    ));
                }
                let (got_plane, got_alpha) =
                    bytes_of(&lane, (&t, surface.as_ref()), layout, &table);
                for (row, _) in &made {
                    found.drawn += 1;
                    let at = row.config * BAND;
                    let band = got_plane
                        .get(at * layout.bytes()..(at + BAND) * layout.bytes())
                        .unwrap_or_default();
                    let normalized;
                    let band = match bits_kept {
                        true => {
                            found.bits_kept += 1;
                            normalized = as_metal_leaves(layout, band);
                            &normalized[..]
                        }
                        false => band,
                    };
                    let alpha_band = got_alpha.get(at..at + BAND).unwrap_or_default();
                    let alpha_digest = match alpha_band.is_empty() {
                        true => 0,
                        false => xr_blends::fnv(alpha_band),
                    };
                    let (p, a) = (xr_blends::fnv(band), alpha_digest);
                    if (p, a) != (row.plane, row.alpha) {
                        let first: String = band
                            .chunks(layout.bytes())
                            .take(3)
                            .map(|t| t.iter().map(|b| format!("{b:02x}")).collect::<String>())
                            .collect::<Vec<_>>()
                            .join(" ");
                        found.wrong.push(format!(
                            "{raw}.{kind} cfg {} {where_:?}: plane {}, alpha {} (first texels {first}, \
                             alpha {:02x?})",
                            row.config,
                            if p == row.plane { "Metal's" } else { "differs" },
                            if a == row.alpha { "Metal's" } else { "differs" },
                            alpha_band.get(..4).unwrap_or_default()
                        ));
                    }
                }
                for one in &announced {
                    let notices = draw(std::slice::from_ref(one), "an announced band");
                    if !notices.contains_key(&source) || notices.contains_key(&other) {
                        found.wrong.push(format!(
                            "{raw}.{kind} cfg {} {where_:?}: announced {notices:?}, the table \
                             says {source:?}",
                            one.0.config
                        ));
                    }
                }
            }
        }
    }
    for line in &found.wrong {
        println!("XR-TABLE-DIFFERS {name} {math:?}: {line}");
    }
    println!(
        "XR-TABLE {name} {math:?}: formats {formats:?} into {lies:?}: {} configurations checked, {} \
         bands drawn ({} in passes announcing extended-range-bits-kept), {} differ; the longest \
         pass of {PASS_BANDS} bands {} ms",
        found.checked,
        found.drawn,
        found.bits_kept,
        found.wrong.len(),
        found.slowest.as_millis()
    );
    assert!(
        found.wrong.is_empty(),
        "{} differ from the table",
        found.wrong.len()
    );
    quiet(lane, &messages);
}

/// `BGRA10_XR` and its `_sRGB` form, into a texture of the device's own and a heap's.
#[test]
fn the_blend_table_holds_for_bgra10_xr() {
    check(&[552, 553], &[Lies::Own, Lies::Heap], Math::Fast);
}

/// `BGR10_XR` and its `_sRGB` form, into a texture of the device's own.
#[test]
fn the_blend_table_holds_for_bgr10_xr() {
    check(&[554, 555], &[Lies::Own], Math::Fast);
}

/// `RGB10A8_2P_XR10`, into a texture of the device's own, a heap's, and one over a surface.
#[test]
fn the_blend_table_holds_for_rgb10a8_2p_xr10() {
    check(&[550], &[Lies::Own, Lies::Heap, Lies::Surface], Math::Fast);
}

/// `RGB10A8_2P_XR10_sRGB`, into a texture of the device's own and one over a surface.
#[test]
fn the_blend_table_holds_for_rgb10a8_2p_xr10_srgb() {
    check(&[551], &[Lies::Own, Lies::Surface], Math::Fast);
}

/// Without fast math, `BGRA10_XR` and its `_sRGB` form, into a texture of the device's own.
#[test]
fn the_blend_table_without_fast_math_holds_for_bgra10_xr() {
    check(&[552, 553], &[Lies::Own], Math::Safe);
}

/// Without fast math, `BGR10_XR` and its `_sRGB` form, into a texture of the device's own.
#[test]
fn the_blend_table_without_fast_math_holds_for_bgr10_xr() {
    check(&[554, 555], &[Lies::Own], Math::Safe);
}

/// Without fast math, `RGB10A8_2P_XR10` and its `_sRGB` form, into a texture of the device's own.
#[test]
fn the_blend_table_without_fast_math_holds_for_rgb10a8_2p_xr10() {
    check(&[550, 551], &[Lies::Own], Math::Safe);
}

/// The second source's blends without fast math, `BGRA10_XR` and its `_sRGB` form, into a texture
/// of the device's own.
#[test]
fn the_second_source_blend_table_without_fast_math_holds_for_bgra10_xr() {
    check(&[552, 553], &[Lies::Own], Math::Second);
}

/// The second source's blends without fast math, `BGR10_XR` and its `_sRGB` form, into a texture of
/// the device's own.
#[test]
fn the_second_source_blend_table_without_fast_math_holds_for_bgr10_xr() {
    check(&[554, 555], &[Lies::Own], Math::Second);
}

/// The second source's blends without fast math, `RGB10A8_2P_XR10` and its `_sRGB` form, into a
/// texture of the device's own.
#[test]
fn the_second_source_blend_table_without_fast_math_holds_for_rgb10a8_2p_xr10() {
    check(&[550, 551], &[Lies::Own], Math::Second);
}

/// A function writing a second output (index 1) beside its first, drawn with blending disabled into
/// `BGRA10_XR` and its `_sRGB` form, a texture of the device's own: the second output stored
/// nowhere, Metal's bytes.
#[test]
fn the_second_output_unblended_holds_for_bgra10_xr() {
    check(&[552, 553], &[Lies::Own], Math::SecondUnblended);
}

/// The same into `BGR10_XR` and its `_sRGB` form.
#[test]
fn the_second_output_unblended_holds_for_bgr10_xr() {
    check(&[554, 555], &[Lies::Own], Math::SecondUnblended);
}

/// The same into `RGB10A8_2P_XR10` and its `_sRGB` form.
#[test]
fn the_second_output_unblended_holds_for_rgb10a8_2p_xr10() {
    check(&[550, 551], &[Lies::Own], Math::SecondUnblended);
}

/// A computed `float4` output under fast math, `BGRA10_XR` and its `_sRGB` form, into a texture of
/// the device's own: the computed class's unkeyed blends are drawn, the others announced.
#[test]
fn the_computed_blend_table_holds_for_bgra10_xr() {
    check(&[552, 553], &[Lies::Own], Math::Computed);
}

/// A computed `float4` output under fast math, `BGR10_XR` and its `_sRGB` form, into a texture of
/// the device's own.
#[test]
fn the_computed_blend_table_holds_for_bgr10_xr() {
    check(&[554, 555], &[Lies::Own], Math::Computed);
}

/// A computed `float4` output under fast math, `RGB10A8_2P_XR10` and its `_sRGB` form, into a
/// texture of the device's own.
#[test]
fn the_computed_blend_table_holds_for_rgb10a8_2p_xr10() {
    check(&[550, 551], &[Lies::Own], Math::Computed);
}

// ---- The memoryless color 2 ----
/// A pipeline of the fixture `xrb_color2`'s `fragment` (in `library`) over color 0 of `pf` under
/// no channel where `copies` is false (and color 2 as `color2`), or every channel and color 2
/// under none where it is true.
fn color2_pipeline(
    (lane, library): (&Lane, &Library),
    pf: PixelFormat,
    fragment: &str,
    (copies, color2): (bool, ColorAttachmentDescriptor),
) -> Result<RenderPipelineState, Error> {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("xrb_color2_vertex"));
    descriptor.fragment = Some(function(fragment));
    let mut color0 = ColorAttachmentDescriptor::new(pf);
    let mut color2 = color2;
    match copies {
        true => color2.write_mask = ColorWriteMask::NONE,
        false => color0.write_mask = ColorWriteMask::NONE,
    }
    descriptor.color_attachments[0] = Some(color0);
    descriptor.color_attachments[2] = Some(color2);
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A two-plane texel's bytes as a fetch of it written back unblended stores them: each color code
/// past 1022 read as 1022, the top two bits zero, the alpha byte as it is.
fn fetched_back(word: &[u8]) -> [u8; 4] {
    let w = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    let code = |shift: u32| ((w >> shift) & 0x3FF).min(1022) << shift;
    (code(20) | code(10) | code(0)).to_le_bytes()
}

/// The runs of the memoryless color 2's pass.
const RUNS: usize = 5;

/// One run of the pass of `descriptor`, `body` encoding it, as [`Lane::run`] runs one: its GPU
/// time in milliseconds (`gpu_end_time` less `gpu_start_time`).
fn timed_run(
    lane: &Lane,
    descriptor: &RenderPassDescriptor,
    body: impl FnOnce(&mut RenderCommandEncoder),
) -> Result<f64, Error> {
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(descriptor)?;
    body(&mut encoder);
    encoder.end_encoding()?;
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Ok((command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?) * 1e3)
}

/// `RGB10A8_2P_XR10`'s blends at color 2, a memoryless attachment loaded and stored `DontCare`
/// (the shape of the consumer's pipelines that blend at color 2), each made configuration of the
/// table's `half4` rows in its band: the fixture `xrb_color2` sets color 2 to the tile value of
/// the table's texels, blends the table's outputs into each band, and copies color 2 to the
/// stored color 0 through a fetch. Each band then holds what the same blend stores into a
/// stored color 0 (drawn first, its bands Metal's digests), read through the same fetch and
/// unblended write ([`fetched_back`]): Metal's memoryless attachment blends as a stored one does
/// (a probe).
#[test]
fn a_memoryless_color_2_blends_as_a_stored_color_does() {
    use super::extended_range_blends::tile_of;
    use super::extended_range_targets::half_bits;
    use super::memoryless::memoryless_options;
    use crate::air_library::AirLibrary;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble};
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let table = xr_blends::table();
    let raw = 550;
    let pf = PixelFormat::try_from(raw).unwrap();
    let layout = Layout::Planes;
    let (plane, alpha) = xr_blends::texels(raw);
    let fixture = library(&lane.device).unwrap();
    let bitcode = llvm_assemble(&engine_dir().join("xrb_color2.ll")).unwrap();
    let color2 = lane.device.new_library_with_air(&bitcode).unwrap();
    let rows: Vec<&xr_blends::Row> = table
        .rows
        .iter()
        .filter(|r| r.format == raw && r.half && r.ours == "made")
        .collect();
    let mut made = Vec::new();
    let mut blended = Vec::new();
    for row in &rows {
        let c = &table.configs[row.config];
        match (
            pipeline((&lane, &fixture, Math::Fast), pf, true, c),
            color2_pipeline(
                (&lane, &color2),
                pf,
                "xrb_color2_fill",
                (false, attachment(pf, c)),
            ),
        ) {
            (Ok(p), Ok(q)) => {
                made.push((*row, p));
                blended.push((*row, q));
            }
            (p, q) => panic!(
                "cfg {}: color 0 {:?}, color 2 {:?}",
                row.config,
                p.err(),
                q.err()
            ),
        }
    }
    assert!(!made.is_empty(), "the table makes no blend at color 2");
    let (w, h) = (WIDTH as u64, height(&table));
    let values = buffer_with(
        &lane.device,
        &xr_blends::sources(true).repeat(table.configs.len()),
    )
    .unwrap();
    let fill_shape = shape(&lane.device, WIDTH as u32, 1).unwrap();
    // The stored color 0, drawn as the table's lane draws it.
    let (stored, _) = target(&lane, pf, Lies::Own, (&plane, &alpha), &table);
    let mut pass = RenderPassDescriptor::new();
    let a = &mut pass.color_attachments[0];
    a.texture = Some(stored.clone());
    a.load_action = LoadAction::Load;
    a.store_action = StoreAction::Store;
    lane.run(&pass, |_, e, _| {
        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&fill_shape), 0, 1).unwrap();
        blend_bands(e, &made, &table);
    })
    .unwrap();
    let (stored_plane, stored_alpha) = bytes_of(&lane, (&stored, None), layout, &table);
    for (row, _) in &made {
        let at = row.config * BAND;
        let band = &stored_plane[4 * at..4 * (at + BAND)];
        let alpha_band = &stored_alpha[at..at + BAND];
        assert_eq!(
            (xr_blends::fnv(band), xr_blends::fnv(alpha_band)),
            (row.plane, row.alpha),
            "cfg {}: the stored color 0 is not Metal's",
            row.config
        );
    }
    // The memoryless color 2: the table's texels' tile values, the blends, the copy.
    let tiles: Vec<u8> = plane
        .chunks(4)
        .zip(&alpha)
        .flat_map(|(word, &a)| {
            tile_of(u32::from_le_bytes([word[0], word[1], word[2], word[3]]), a).map(half_bits)
        })
        .flat_map(u16::to_le_bytes)
        .collect();
    let tiles = buffer_with(&lane.device, &tiles.repeat(table.configs.len())).unwrap();
    let fill = color2_pipeline(
        (&lane, &color2),
        pf,
        "xrb_color2_fill",
        (false, ColorAttachmentDescriptor::new(pf)),
    )
    .unwrap();
    let copy = color2_pipeline(
        (&lane, &color2),
        pf,
        "xrb_color2_copy",
        (true, ColorAttachmentDescriptor::new(pf)),
    )
    .unwrap();
    let (color0, _) = target(&lane, pf, Lies::Own, (&plane, &alpha), &table);
    let mut descriptor = TextureDescriptor::texture_2d(pf, w, h, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = memoryless_options();
    let scratch = lane.device.new_texture(&descriptor).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let a = &mut pass.color_attachments[0];
    a.texture = Some(color0.clone());
    a.load_action = LoadAction::DontCare;
    a.store_action = StoreAction::Store;
    let a = &mut pass.color_attachments[2];
    a.texture = Some(scratch);
    a.load_action = LoadAction::DontCare;
    a.store_action = StoreAction::DontCare;
    let whole = ScissorRect::new(0, 0, w, h);
    // The pass runs `RUNS` times, each checked: a device that orders its draws wrong may do so in
    // some runs alone, in some bands (RADV on an AMD 780M before the storage barrier). Each run's
    // GPU time is printed.
    let mut wrong = Vec::new();
    let mut times = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let time = timed_run(&lane, &pass, |e| {
            e.set_fragment_buffer(Some(&fill_shape), 0, 1).unwrap();
            e.set_fragment_buffer(Some(&tiles), 0, 0).unwrap();
            e.set_render_pipeline_state(&fill).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
            blend_bands(e, &blended, &table);
            e.set_scissor_rect(whole).unwrap();
            e.set_render_pipeline_state(&copy).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        })
        .unwrap_or_else(|e| panic!("the pass of the memoryless color 2: {e}"));
        times.push(format!("{time:.3}"));
        let (got_plane, got_alpha) = bytes_of(&lane, (&color0, None), layout, &table);
        for (row, _) in &made {
            let at = row.config * BAND;
            let want: Vec<u8> = stored_plane[4 * at..4 * (at + BAND)]
                .chunks(4)
                .flat_map(fetched_back)
                .collect();
            let got = &got_plane[4 * at..4 * (at + BAND)];
            let differ: Vec<usize> = (0..BAND)
                .filter(|&i| {
                    got[4 * i..4 * i + 4] != want[4 * i..4 * i + 4]
                        || got_alpha[at + i] != stored_alpha[at + i]
                })
                .collect();
            if !differ.is_empty() {
                let word = |b: &[u8], i: usize| {
                    u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
                };
                let first: Vec<String> = differ
                    .iter()
                    .take(3)
                    .map(|&i| {
                        format!(
                            "texel {i} ({}, {}): {:08x}/{:02x}, want {:08x}/{:02x}",
                            i % WIDTH,
                            row.config * BAND_HEIGHT + i / WIDTH,
                            word(got, i),
                            got_alpha[at + i],
                            word(&want, i),
                            stored_alpha[at + i]
                        )
                    })
                    .collect();
                wrong.push(format!(
                    "run {run}: cfg {}: {} of {BAND} texels ({})",
                    row.config,
                    differ.len(),
                    first.join("; ")
                ));
            }
        }
    }
    for line in &wrong {
        println!("XR-TABLE-DIFFERS {name}: color 2: {line}");
    }
    println!(
        "XR-TABLE {name}: color 2 memoryless: {} blends drawn in each of {RUNS} runs, {} differ \
         from the stored color's; GPU ms per run: {}",
        made.len(),
        wrong.len(),
        times.join(" ")
    );
    assert!(wrong.is_empty(), "{} blends at color 2 differ", wrong.len());
    quiet(lane, &messages);
}
