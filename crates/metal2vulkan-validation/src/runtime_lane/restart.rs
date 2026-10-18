//! Device tests of primitive restart on the device the cfg selects, under the validation layer. The
//! owned fixture `rt6_restart` places vertex `v` at the position buffer 0 holds at `min(v, 65535)`,
//! and its fragment function adds one 8-bit unit through an additive blend, so each pixel of an 8x1
//! `RGBA8Unorm` target cleared to zero counts the primitives that cover it.
//!
//! Each case's counts are those Metal stored (a Metal probe on an M4 with macOS 27.0.1, with and
//! without API validation): an indexed strip of triangles or lines restarts at the all-ones index
//! of its index type, `ushort` and `uint`, and a list of triangles or lines, or points, draws the
//! all-ones index as a vertex (the fixture places a `uint` all-ones where vertex 65535 is). Another
//! index splits nothing.
//!
//! After `set_primitive_restart_enabled` (Apple's `setPrimitiveRestartEnabled:index:`), an indexed
//! strip restarts at the call's index `k'` alone (an M4 with macOS 27.0.1): `k mod 2^32` for 32-bit
//! indices, the lesser of that and `0xffff` for 16-bit ones, compared with the index before the
//! base vertex; the all-ones index then draws as a vertex; each piece starts its winding afresh;
//! lists and points do not restart; the call lasts until the encoder ends. Those cases draw on a
//! 16x16 target, over scattered positions, and each must count what the same device counts for
//! Metal's answer drawn plainly: each piece of the strip a strip of its own, of 32-bit indices that
//! no piece restarts in (the all-ones index as vertex 65535, which the fixture's clamp makes the
//! same place). They cover direct and indirect draws, 16- and 32-bit indices, lines and triangles,
//! a base vertex, positions past the index buffer (index 0, restarting where `k'` is 0), indices a
//! blit wrote earlier in the command buffer, culled windings, the call's lifetime, and a last
//! 16-bit index alone in its word (an odd count, or an offset 2 bytes past a multiple of 4).

use super::{buffer_with, dynamic_input_assembly, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::{
    BlendFactor, CullMode, IndexType, PixelFormat, PrimitiveType, TextureUsage, Winding,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::ApiVersion;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Device, Error,
    LoadAction, PipelineOptions, Region, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, StoreAction, Texture, TextureDescriptor,
};

/// The all-ones index of each index type.
const ONES16: u32 = 0xffff;
const ONES32: u32 = u32::MAX;

/// Two strips split by each index type's all-ones index; and a list with it.
const STRIP16: &[u32] = &[0, 1, 2, ONES16, 3, 4, 5];
const STRIP32: &[u32] = &[0, 1, 2, ONES32, 3, 4, 5];
const LIST16: &[u32] = &[0, 1, 2, ONES16, 6, 7];

/// A draw: its primitive type, index type and indices, and the counts Metal stored, one digit
/// per pixel.
struct Case {
    name: &'static str,
    primitive: PrimitiveType,
    index_type: IndexType,
    indices: &'static [u32],
    metal: &'static str,
}

impl Case {
    /// Whether the case draws on the lines' and points' positions, else the triangles'.
    fn on_the_row(&self) -> bool {
        !matches!(
            self.primitive,
            PrimitiveType::Triangle | PrimitiveType::TriangleStrip
        )
    }

    /// The index buffer's bytes.
    fn index_bytes(&self) -> Vec<u8> {
        if self.index_type == IndexType::UInt16 {
            self.indices
                .iter()
                .map(|&i| u16::try_from(i).unwrap())
                .flat_map(u16::to_le_bytes)
                .collect()
        } else {
            self.indices.iter().flat_map(|i| i.to_le_bytes()).collect()
        }
    }

    /// The counts Metal stored.
    fn metal(&self) -> [u8; 8] {
        let digits = self.metal.as_bytes();
        std::array::from_fn(|pixel| digits[pixel] - b'0')
    }
}

/// The draws of probe 15c.
fn cases() -> Vec<Case> {
    use IndexType::{UInt16, UInt32};
    use PrimitiveType::{Line, LineStrip, Point, Triangle, TriangleStrip};
    let case = |name, primitive, index_type, indices, metal| Case {
        name,
        primitive,
        index_type,
        indices,
        metal,
    };
    vec![
        case("strip16", TriangleStrip, UInt16, STRIP16, "10000001"),
        case("strip32", TriangleStrip, UInt32, STRIP32, "10000001"),
        case(
            "strip16_whole",
            TriangleStrip,
            UInt16,
            &[0, 1, 2, 65534, 3, 4, 5],
            "11112112",
        ),
        case("list16", Triangle, UInt16, LIST16, "10001000"),
        case(
            "list32",
            Triangle,
            UInt32,
            &[0, 1, 2, ONES32, 6, 7],
            "10001000",
        ),
        case(
            "linestrip16",
            LineStrip,
            UInt16,
            &[8, 9, ONES16, 10, 11],
            "10000010",
        ),
        case(
            "linestrip32",
            LineStrip,
            UInt32,
            &[8, 9, ONES32, 10, 11],
            "10000010",
        ),
        case("lines16", Line, UInt16, &[8, 9, ONES16, 11], "10001110"),
        case("lines32", Line, UInt32, &[8, 9, ONES32, 11], "10001110"),
        case("points16", Point, UInt16, &[12, ONES16, 13], "10001001"),
        case("points32", Point, UInt32, &[12, ONES32, 13], "10001001"),
    ]
}

/// The center of pixel `i` of the row, in normalized device coordinates.
fn center(i: u8) -> f32 {
    -1.0 + f32::from(2 * i + 1) / 8.0
}

/// The fixture's 65536 positions, `(x, y)` each, as probe 15c wrote them. The triangles': strip
/// A covers pixel 0 (vertices 0 to 2), strip B pixel 7 (3 to 5), and vertices 6 and 7 with vertex
/// 65535 pixel 4. The lines' and points': on the row, vertex 65535 at pixel 4. Every other
/// vertex lies off the target, at (5, 5).
fn positions(on_the_row: bool) -> Vec<u8> {
    let mut all = vec![5.0f32; 65536 * 2];
    let set: &[(usize, (f32, f32))] = if on_the_row {
        &[
            (8, (center(0), 0.0)),
            (9, (center(1), 0.0)),
            (10, (center(6), 0.0)),
            (11, (center(7), 0.0)),
            (12, (center(0), 0.0)),
            (13, (center(7), 0.0)),
            (65535, (center(4), 0.0)),
        ]
    } else {
        &[
            (0, (center(0) - 0.1, -1.0)),
            (1, (center(0) + 0.1, -1.0)),
            (2, (center(0), 1.0)),
            (3, (center(7) - 0.1, -1.0)),
            (4, (center(7) + 0.1, -1.0)),
            (5, (center(7), 1.0)),
            (6, (center(4) - 0.1, -1.0)),
            (7, (center(4) + 0.1, -1.0)),
            (65535, (center(4), 1.0)),
        ]
    };
    for &(vertex, (x, y)) in set {
        all[2 * vertex] = x;
        all[2 * vertex + 1] = y;
    }
    all.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The lane's device, its queue, and the fixture's pipeline: additive blending into
/// `RGBA8Unorm`.
struct Lane {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("rt6_restart.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name| {
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function("rt6_restart_vertex")));
        descriptor.fragment = Some(StageSource::Function(function("rt6_restart_count")));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::One;
        color.destination_rgb_blend_factor = BlendFactor::One;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_alpha_blend_factor = BlendFactor::One;
        descriptor.color_attachments[0] = Some(color);
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            pipeline,
            device,
        }
    }

    /// Draw `case` into a cleared 8x1 target, and return each pixel's count (its red channel).
    fn counts(&self, case: &Case) -> Result<[u8; 8], Error> {
        self.counts_of(&[case])
    }

    /// Draw `cases`, of one row's positions, back to back in one encoder into a cleared 8x1
    /// target, and return each pixel's count: the sum of the draws'.
    fn counts_of(&self, cases: &[&Case]) -> Result<[u8; 8], Error> {
        let on_the_row = cases[0].on_the_row();
        assert!(cases.iter().all(|case| case.on_the_row() == on_the_row));
        let positions = buffer_with(&self.device, &positions(on_the_row)).unwrap();
        let index_buffers: Vec<Buffer> = cases
            .iter()
            .map(|case| buffer_with(&self.device, &case.index_bytes()).unwrap())
            .collect();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 1, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&target).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::Store;
        attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(&self.pipeline)?;
        encoder.set_vertex_buffer(Some(&positions), 0, 0)?;
        let drawn = cases
            .iter()
            .zip(&index_buffers)
            .try_for_each(|(case, index_buffer)| {
                encoder.draw_indexed_primitives(
                    case.primitive,
                    case.indices.len() as u32,
                    case.index_type,
                    index_buffer,
                    0,
                    1,
                    0,
                    0,
                )
            });
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drawn?;
        let mut texels = [0u8; 32];
        target.get_bytes(&mut texels, 32, 32, Region::new_2d(0, 0, 8, 1), 0, 0)?;
        Ok(std::array::from_fn(|pixel| texels[4 * pixel]))
    }
}

/// Every case counts what Metal counted: an indexed strip restarts at its index type's all-ones
/// index and nowhere else, and a list or points draw it as a vertex.
#[test]
fn indexed_strips_restart_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        for case in cases() {
            let counts = lane
                .counts(&case)
                .unwrap_or_else(|e| panic!("{} on {name}: {e}", case.name));
            assert_eq!(counts, case.metal(), "{} on {name}: the counts", case.name);
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// A strip, a list of its class and the strip again, drawn back to back in one encoder through
/// the lane's one state, count the sum of what Metal counted for each: each draw restarts as its
/// own type does, the strip at the all-ones index and the list nowhere, whether the pipeline
/// holds the restart statically or each draw records it. The device makes the topology and
/// restart dynamic exactly where it has both extended dynamic states (core from Vulkan 1.3, else
/// `VK_EXT_extended_dynamic_state` and `VK_EXT_extended_dynamic_state2` with their features, as
/// the physical device lists them): there a strip takes the pipeline its class's list made, so
/// the strips make none; elsewhere the first strip makes its own. Where the driver is measured
/// to warn of such a pipeline (`DriverFact::DynamicRestartWarns`: MoltenVK 1.4.1), the warning
/// arrives at each such pipeline's creation and is withheld there, so the count of withheld
/// diagnostics grows and the layer reports nothing; elsewhere none is withheld.
#[test]
fn a_strip_and_a_list_of_one_class_restart_as_metal_in_one_encoder() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let dynamic = dynamic_input_assembly(&device);
        assert_eq!(
            dynamic,
            has_both_extended_dynamic_states(&device),
            "{name} (API {:?}): the runtime makes the input assembly dynamic exactly where the \
             device has both extended dynamic states",
            device.properties().api_version
        );
        eprintln!("{name}: dynamic topology and restart {dynamic}");
        let warns = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DynamicRestartWarns);
        let lane = Lane::new(device);
        let all = cases();
        let case = |name: &str| all.iter().find(|case| case.name == name).unwrap();
        for (strip, list) in [("strip16", "list16"), ("linestrip16", "lines16")] {
            let (strip, list) = (case(strip), case(list));
            // The list's pipeline first; then what the strips make.
            lane.counts(list).unwrap();
            let before = lane.device.work_counts();
            let counts = lane
                .counts_of(&[strip, list, strip])
                .unwrap_or_else(|e| panic!("{} and {} on {name}: {e}", strip.name, list.name));
            let made =
                lane.device.work_counts().variant_pipelines_made - before.variant_pipelines_made;
            let (strip_counts, list_counts) = (strip.metal(), list.metal());
            let metal: [u8; 8] =
                std::array::from_fn(|pixel| 2 * strip_counts[pixel] + list_counts[pixel]);
            assert_eq!(
                counts, metal,
                "{} and {} on {name}: the counts",
                strip.name, list.name
            );
            assert_eq!(
                made,
                u64::from(!dynamic),
                "{} after {} on {name} (dynamic topology {dynamic}): pipelines made",
                strip.name,
                list.name
            );
        }
        let withheld = lane.device.work_counts().driver_diagnostics_withheld;
        eprintln!("{name}: driver diagnostics withheld {withheld}");
        assert_eq!(
            withheld > 0,
            dynamic && warns,
            "{name}: {withheld} driver diagnostics withheld (dynamic restart {dynamic}, the \
             driver measured to warn of it {warns})"
        );
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// Whether `device`'s physical device, as its instance sees it, has both extended dynamic states:
/// at an effective Vulkan 1.3 or later, where both are core without a feature; below it, where
/// it lists `VK_EXT_extended_dynamic_state` and `VK_EXT_extended_dynamic_state2` and reports
/// `extendedDynamicState` and `extendedDynamicState2`. Read through Vulkan itself, not the
/// runtime's description, so a runtime that failed to enable them is caught.
fn has_both_extended_dynamic_states(device: &Device) -> bool {
    use ash::vk;
    if device.properties().api_version >= ApiVersion::V1_3 {
        return true;
    }
    // SAFETY: the handles are used only here, while the device lives, to query the physical
    // device.
    let raw = unsafe { device.raw() };
    // SAFETY: a physical device of the instance.
    let listed = unsafe {
        raw.instance
            .enumerate_device_extension_properties(raw.physical_device)
    }
    .expect("the device's extensions");
    let lists = |name: &std::ffi::CStr| {
        listed
            .iter()
            .any(|p| p.extension_name_as_c_str() == Ok(name))
    };
    if !lists(ash::ext::extended_dynamic_state::NAME)
        || !lists(ash::ext::extended_dynamic_state2::NAME)
    {
        return false;
    }
    let mut first = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default();
    let mut second = vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut first)
        .push_next(&mut second);
    // SAFETY: a physical device of the instance, which lists both structures' extensions.
    unsafe {
        raw.instance
            .get_physical_device_features2(raw.physical_device, &mut features)
    };
    first.extended_dynamic_state == vk::TRUE && second.extended_dynamic_state2 == vk::TRUE
}

/// The side of the target the chosen-index cases count on.
const SIDE: u32 = 16;

/// The vertices the chosen-index cases scatter, besides vertex 65535.
const SCATTERED: u32 = 64;

/// The fixture's 65536 positions for the chosen-index cases: vertices 0 to 63 and 65535 at
/// points of a fixed pseudo-random walk over the target and a little past it, every other vertex
/// off the target, at (5, 5). Lines draw between pixel centers.
fn scattered(lines: bool) -> Vec<u8> {
    let mut all = vec![5.0f32; 65536 * 2];
    let mut state = 0x2545_f491_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let side = SIDE as f32;
    for vertex in (0..SCATTERED as usize).chain([65535]) {
        let (x, y) = if lines {
            let pixel = |n: u32| -1.0 + (2.0 * (n % SIDE) as f32 + 1.0) / side;
            (pixel(next()), pixel(next()))
        } else {
            let unit = |n: u32| (n % 10_000) as f32 / 10_000.0 * 2.5 - 1.25;
            (unit(next()), unit(next()))
        };
        all[2 * vertex] = x;
        all[2 * vertex + 1] = y;
    }
    all.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `values` as indices of `index_type`.
fn bytes_of(index_type: IndexType, values: &[u32]) -> Vec<u8> {
    match index_type {
        IndexType::UInt16 => values
            .iter()
            .map(|&i| u16::try_from(i).unwrap())
            .flat_map(u16::to_le_bytes)
            .collect(),
        _ => values.iter().flat_map(|i| i.to_le_bytes()).collect(),
    }
}

/// How a chosen-index case draws.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    /// `draw_indexed_primitives`.
    Direct,
    /// `draw_indexed_primitives_indirect`, its arguments' index start this many indices in (the
    /// index buffer holds that many indices of 0 before the case's).
    Indirect(u32),
    /// `draw_indexed_primitives` over an index buffer holding other indices, onto which a blit
    /// earlier in the command buffer copies the case's.
    Written,
}

/// A chosen-index case: one draw after one call.
struct Chosen {
    name: &'static str,
    primitive: PrimitiveType,
    index_type: IndexType,
    /// The index buffer's indices.
    indices: Vec<u32>,
    /// The positions the draw draws: the indices', and index 0 past them.
    count: u32,
    /// The encoder's call before the draw.
    k: u64,
    base_vertex: i32,
    form: Form,
    /// Back faces culled, counter-clockwise ones in front.
    culled: bool,
    /// The draw's index buffer offset in bytes, a multiple of the index size: the index buffer
    /// holds that many bytes of other indices before the case's.
    offset: u64,
}

impl Chosen {
    /// The index Metal restarts the case's strip at (`k'`), `None` for a list or points.
    fn restart(&self) -> Option<u32> {
        let strip = matches!(
            self.primitive,
            PrimitiveType::LineStrip | PrimitiveType::TriangleStrip
        );
        let low = self.k as u32;
        strip.then_some(match self.index_type {
            IndexType::UInt16 => low.min(0xffff),
            _ => low,
        })
    }

    /// Metal's answer as plain draws: each piece of the positions the draw reads, split where
    /// the index is `k'`, as 32-bit indices, the all-ones index as vertex 65535.
    fn pieces(&self) -> Vec<Vec<u32>> {
        let positions = (0..self.count as usize).map(|p| self.indices.get(p).copied().unwrap_or(0));
        let mut pieces = vec![Vec::new()];
        for index in positions {
            if Some(index) == self.restart() {
                pieces.push(Vec::new());
            } else {
                pieces.last_mut().unwrap().push(index.min(65535));
            }
        }
        pieces.retain(|piece| !piece.is_empty());
        pieces
    }
}

/// The chosen-index cases.
fn chosen() -> Vec<Chosen> {
    use IndexType::{UInt16, UInt32};
    use PrimitiveType::{Line, LineStrip, Point, Triangle, TriangleStrip};
    let walk = |n: u32, step: u32| -> Vec<u32> { (0..n).map(|i| (i * step) % SCATTERED).collect() };
    // A strip of 30 vertices with restarts at 7 and the all-ones indices among them.
    let mut strip = walk(30, 5);
    strip[4] = 7;
    strip[11] = 7;
    strip[12] = 7;
    strip[17] = 7;
    let with = |mut indices: Vec<u32>, at: &[usize], value: u32| {
        for &p in at {
            indices[p] = value;
        }
        indices
    };
    let strip16 = with(strip.clone(), &[8, 22], 0xffff);
    let strip32 = with(strip.clone(), &[8, 22], u32::MAX);
    let case = |name, primitive, index_type, indices: Vec<u32>, k| Chosen {
        name,
        primitive,
        index_type,
        count: indices.len() as u32,
        indices,
        k,
        base_vertex: 0,
        form: Form::Direct,
        culled: false,
        offset: 0,
    };
    let mut cases = vec![
        case("tri16_k7", TriangleStrip, UInt16, strip16.clone(), 7),
        case("tri32_k7", TriangleStrip, UInt32, strip32.clone(), 7),
        case(
            "tri16_k_past_32_bits",
            TriangleStrip,
            UInt16,
            strip16.clone(),
            0x1_0000_0007,
        ),
        case(
            "tri32_k_past_32_bits",
            TriangleStrip,
            UInt32,
            strip32.clone(),
            0x7_0000_0007,
        ),
        case(
            "tri16_k_past_16_bits",
            TriangleStrip,
            UInt16,
            strip16.clone(),
            0x1_0007,
        ),
        case(
            "tri32_k_all_ones",
            TriangleStrip,
            UInt32,
            strip32.clone(),
            u64::MAX,
        ),
        case("line16_k7", LineStrip, UInt16, strip16.clone(), 7),
        case("line32_k7", LineStrip, UInt32, strip32.clone(), 7),
        case("list16_k7", Triangle, UInt16, strip16[..27].to_vec(), 7),
        case("lines32_k7", Line, UInt32, strip32[..28].to_vec(), 7),
        case("points16_k7", Point, UInt16, strip16.clone(), 7),
    ];
    let mut based = case("tri16_k7_based", TriangleStrip, UInt16, strip16.clone(), 7);
    based.base_vertex = 3;
    cases.push(based);
    let mut past = case("tri32_k0_past", TriangleStrip, UInt32, strip32.clone(), 0);
    past.count += 5;
    cases.push(past);
    let mut past = case("tri32_k7_past", TriangleStrip, UInt32, strip32.clone(), 7);
    past.count += 5;
    cases.push(past);
    let mut culled = case("tri32_k7_culled", TriangleStrip, UInt32, strip32.clone(), 7);
    culled.culled = true;
    cases.push(culled);
    let mut indirect = case(
        "tri16_k7_indirect",
        TriangleStrip,
        UInt16,
        strip16.clone(),
        7,
    );
    indirect.form = Form::Indirect(3);
    indirect.base_vertex = 2;
    cases.push(indirect);
    let mut indirect = case("line32_k7_indirect", LineStrip, UInt32, strip32.clone(), 7);
    indirect.form = Form::Indirect(0);
    cases.push(indirect);
    let mut written = case(
        "tri32_k7_written",
        TriangleStrip,
        UInt32,
        strip32.clone(),
        7,
    );
    written.form = Form::Written;
    cases.push(written);
    let mut written = case(
        "tri16_k7_written_indirect",
        TriangleStrip,
        UInt16,
        strip16,
        7,
    );
    written.form = Form::Indirect(1);
    cases.push(written);
    // The last of the draw's 16-bit indices alone in its word: an odd count from byte 0, or an
    // even one from byte 2 (nine indices restarting at 3, from an 18- or a 20-byte buffer, drew
    // their last triangle to vertex 0 on the M3 Max where ten were exact).
    let mut nine = walk(9, 5);
    nine[3] = 3;
    let mut ten = walk(10, 5);
    ten[3] = 3;
    cases.push(case(
        "tri16_k3_nine",
        TriangleStrip,
        UInt16,
        nine.clone(),
        3,
    ));
    cases.push(case("line16_k3_nine", LineStrip, UInt16, nine.clone(), 3));
    let mut of_ten = case(
        "tri16_k3_nine_of_ten",
        TriangleStrip,
        UInt16,
        ten.clone(),
        3,
    );
    of_ten.count = 9;
    cases.push(of_ten);
    let mut from_2 = case("tri16_k3_ten_from_2", TriangleStrip, UInt16, ten.clone(), 3);
    from_2.offset = 2;
    cases.push(from_2);
    let mut indirect = case("tri16_k3_nine_indirect", TriangleStrip, UInt16, nine, 3);
    indirect.form = Form::Indirect(0);
    cases.push(indirect);
    let mut indirect = case(
        "tri16_k3_ten_from_2_indirect",
        TriangleStrip,
        UInt16,
        ten,
        3,
    );
    indirect.form = Form::Indirect(0);
    indirect.offset = 2;
    cases.push(indirect);
    // Many pieces across the pass's runs: 3000 positions, a restart every 37.
    let long: Vec<u32> = (0..3000u32)
        .map(|i| match i % 37 {
            36 => 7,
            _ => ((i * 11 + 1) % SCATTERED) | 8,
        })
        .collect();
    cases.push(case(
        "tri32_k7_long",
        TriangleStrip,
        UInt32,
        long.clone(),
        7,
    ));
    let mut long = case("line16_k7_long_indirect", LineStrip, UInt16, long, 7);
    long.form = Form::Indirect(0);
    cases.push(long);
    cases
}

impl Lane {
    /// Run `encode` with a fresh command buffer and a pass that clears a `SIDE`x`SIDE` target,
    /// commit it, and return the command buffer and each pixel's count (its red channel).
    fn grid(
        &self,
        encode: impl FnOnce(&CommandBuffer, &RenderPassDescriptor) -> Result<(), Error>,
    ) -> Result<(CommandBuffer, Vec<u8>), Error> {
        let side = u64::from(SIDE);
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let target: Texture = self.device.new_texture(&target).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::Store;
        attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let command_buffer = self.queue.command_buffer()?;
        encode(&command_buffer, &pass)?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let bytes = (4 * SIDE * SIDE) as usize;
        let mut texels = vec![0u8; bytes];
        let row = u64::from(4 * SIDE);
        target.get_bytes(
            &mut texels,
            row,
            bytes as u64,
            Region::new_2d(0, 0, side, side),
            0,
            0,
        )?;
        let counts = texels.chunks(4).map(|texel| texel[0]).collect();
        Ok((command_buffer, counts))
    }

    /// An encoder of `pass` with the fixture's pipeline and `positions`.
    fn encoder(
        &self,
        command_buffer: &CommandBuffer,
        pass: &RenderPassDescriptor,
        positions: &Buffer,
    ) -> Result<RenderCommandEncoder, Error> {
        let mut encoder = command_buffer.render_command_encoder(pass)?;
        encoder.set_render_pipeline_state(&self.pipeline)?;
        encoder.set_vertex_buffer(Some(positions), 0, 0)?;
        Ok(encoder)
    }

    /// Metal's answer to `case`: its pieces drawn as plain strips (or the list or points), each
    /// of 32-bit indices from a buffer of its own, with the case's base vertex and culling.
    fn plainly(&self, case: &Chosen, positions: &Buffer) -> Result<Vec<u8>, Error> {
        let buffers: Vec<(Buffer, u32)> = case
            .pieces()
            .iter()
            .map(|piece| {
                let buffer = buffer_with(&self.device, &bytes_of(IndexType::UInt32, piece));
                (buffer.unwrap(), piece.len() as u32)
            })
            .collect();
        let (command_buffer, counts) = self.grid(|command_buffer, pass| {
            let mut encoder = self.encoder(command_buffer, pass, positions)?;
            cull(&mut encoder, case.culled);
            for (buffer, count) in &buffers {
                encoder.draw_indexed_primitives(
                    case.primitive,
                    *count,
                    IndexType::UInt32,
                    buffer,
                    0,
                    1,
                    case.base_vertex,
                    0,
                )?;
            }
            encoder.end_encoding()
        })?;
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Ok(counts)
    }

    /// `case` drawn after its call, and its command buffer's GPU time in milliseconds.
    fn restarted(&self, case: &Chosen, positions: &Buffer) -> Result<(Vec<u8>, f64), Error> {
        let (lead, arguments) = match case.form {
            Form::Indirect(start) => (start, Some(start)),
            Form::Direct | Form::Written => (0, None),
        };
        let size = match case.index_type {
            IndexType::UInt16 => 2,
            _ => 4,
        };
        let mut values = vec![1; (case.offset / size) as usize];
        values.extend(std::iter::repeat_n(0, lead as usize));
        values.extend(&case.indices);
        let bytes = bytes_of(case.index_type, &values);
        let staging = buffer_with(&self.device, &bytes).unwrap();
        let written = case.form == Form::Written || matches!(case.form, Form::Indirect(1));
        // A written case's index buffer holds a walk the other way first.
        let first: Vec<u32> = match written {
            true => (0..values.len() as u32)
                .map(|i| (i * 3 + 2) % SCATTERED)
                .collect(),
            false => values.clone(),
        };
        let index_buffer = buffer_with(&self.device, &bytes_of(case.index_type, &first)).unwrap();
        let indirect = arguments.map(|start| {
            let words = [case.count, 1, start, case.base_vertex as u32, 0];
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            buffer_with(&self.device, &bytes).unwrap()
        });
        let (command_buffer, counts) = self.grid(|command_buffer, pass| {
            if written {
                let mut blit = command_buffer.blit_command_encoder()?;
                blit.copy_from_buffer_to_buffer(&staging, 0, &index_buffer, 0, bytes.len() as u64)?;
                blit.end_encoding()?;
            }
            let mut encoder = self.encoder(command_buffer, pass, positions)?;
            cull(&mut encoder, case.culled);
            encoder.set_primitive_restart_enabled(true, case.k);
            match &indirect {
                Some(indirect) => encoder.draw_indexed_primitives_indirect(
                    case.primitive,
                    case.index_type,
                    &index_buffer,
                    case.offset,
                    indirect,
                    0,
                )?,
                None => encoder.draw_indexed_primitives(
                    case.primitive,
                    case.count,
                    case.index_type,
                    &index_buffer,
                    case.offset,
                    1,
                    case.base_vertex,
                    0,
                )?,
            }
            encoder.end_encoding()
        })?;
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let gpu = command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?;
        Ok((counts, gpu * 1000.0))
    }

    /// A strip `case`'s pieces drawn as the list the rewrite pass writes, without the pass: one
    /// draw of 32-bit indices, each piece's triangles in Vulkan's strip order (or its segments);
    /// the counts, and the command buffer's GPU time in milliseconds.
    fn listed(&self, case: &Chosen, positions: &Buffer) -> Result<(Vec<u8>, f64), Error> {
        let mut list = Vec::new();
        for piece in case.pieces() {
            match case.primitive {
                PrimitiveType::LineStrip => {
                    for pair in piece.windows(2) {
                        list.extend(pair);
                    }
                }
                _ => {
                    for (i, t) in piece.windows(3).enumerate() {
                        match i % 2 {
                            0 => list.extend([t[0], t[1], t[2]]),
                            _ => list.extend([t[0], t[2], t[1]]),
                        }
                    }
                }
            }
        }
        let primitive = match case.primitive {
            PrimitiveType::LineStrip => PrimitiveType::Line,
            _ => PrimitiveType::Triangle,
        };
        let indices = buffer_with(&self.device, &bytes_of(IndexType::UInt32, &list)).unwrap();
        let (command_buffer, counts) = self.grid(|command_buffer, pass| {
            let mut encoder = self.encoder(command_buffer, pass, positions)?;
            cull(&mut encoder, case.culled);
            encoder.draw_indexed_primitives(
                primitive,
                list.len() as u32,
                IndexType::UInt32,
                &indices,
                0,
                1,
                case.base_vertex,
                0,
            )?;
            encoder.end_encoding()
        })?;
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let gpu = command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?;
        Ok((counts, gpu * 1000.0))
    }
}

/// Cull back faces, counter-clockwise ones in front, where `culled`.
fn cull(encoder: &mut RenderCommandEncoder, culled: bool) {
    if culled {
        encoder.set_cull_mode(CullMode::Back);
        encoder.set_front_facing_winding(Winding::CounterClockwise);
    }
}

/// Every chosen-index case counts what Metal's answer drawn plainly counts on the same device.
#[test]
fn strips_restart_at_the_called_index_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        for case in chosen() {
            let lines = matches!(
                case.primitive,
                PrimitiveType::Line | PrimitiveType::LineStrip
            );
            let positions = buffer_with(&lane.device, &scattered(lines)).unwrap();
            let expected = lane.plainly(&case, &positions).unwrap();
            assert!(
                expected.iter().any(|&count| count > 0),
                "{}: Metal's answer draws nothing",
                case.name
            );
            let counts = lane
                .restarted(&case, &positions)
                .unwrap_or_else(|e| panic!("{} on {name}: {e}", case.name))
                .0;
            assert_eq!(counts, expected, "{} on {name}: the counts", case.name);
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// The call lasts from where it is made, before the pipeline state or after, until the encoder
/// ends, and the last call wins; a draw before it, and the next encoder, restart at the all-ones
/// index.
#[test]
fn the_restart_call_lasts_until_its_encoder_ends() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let positions = buffer_with(&lane.device, &scattered(false)).unwrap();
        let mut indices: Vec<u32> = (0..24).map(|i| (i * 7 + 3) % SCATTERED).collect();
        indices[6] = 9;
        indices[15] = 0xffff;
        let index_buffer =
            buffer_with(&lane.device, &bytes_of(IndexType::UInt16, &indices)).unwrap();
        let at = |k: u64| Chosen {
            name: "lifetime",
            primitive: PrimitiveType::TriangleStrip,
            index_type: IndexType::UInt16,
            count: indices.len() as u32,
            indices: indices.clone(),
            k,
            base_vertex: 0,
            form: Form::Direct,
            culled: false,
            offset: 0,
        };
        let at_nine = lane.plainly(&at(9), &positions).unwrap();
        let at_ones = lane.plainly(&at(0xffff), &positions).unwrap();
        assert_ne!(at_nine, at_ones, "the cases tell the indices apart");
        let draw = |encoder: &mut RenderCommandEncoder| {
            encoder.draw_indexed_primitives(
                PrimitiveType::TriangleStrip,
                indices.len() as u32,
                IndexType::UInt16,
                &index_buffer,
                0,
                1,
                0,
                0,
            )
        };
        type Sequence<'a> = &'a dyn Fn(&CommandBuffer, &RenderPassDescriptor) -> Result<(), Error>;
        let before_the_pipeline: Sequence = &|command_buffer, pass| {
            let mut encoder = command_buffer.render_command_encoder(pass)?;
            encoder.set_primitive_restart_enabled(false, 9);
            encoder.set_render_pipeline_state(&lane.pipeline)?;
            encoder.set_vertex_buffer(Some(&positions), 0, 0)?;
            draw(&mut encoder)?;
            encoder.end_encoding()
        };
        let last_wins: Sequence = &|command_buffer, pass| {
            let mut encoder = lane.encoder(command_buffer, pass, &positions)?;
            encoder.set_primitive_restart_enabled(true, 3);
            encoder.set_primitive_restart_enabled(true, 9);
            draw(&mut encoder)?;
            encoder.end_encoding()
        };
        let after_the_draw: Sequence = &|command_buffer, pass| {
            let mut encoder = lane.encoder(command_buffer, pass, &positions)?;
            draw(&mut encoder)?;
            encoder.set_primitive_restart_enabled(true, 9);
            encoder.end_encoding()
        };
        let next_encoder: Sequence = &|command_buffer, pass| {
            let mut encoder = lane.encoder(command_buffer, pass, &positions)?;
            encoder.set_primitive_restart_enabled(true, 9);
            encoder.end_encoding()?;
            let mut encoder = lane.encoder(command_buffer, pass, &positions)?;
            draw(&mut encoder)?;
            encoder.end_encoding()
        };
        for (sequence, encode, expected) in [
            ("before_the_pipeline", before_the_pipeline, &at_nine),
            ("last_wins", last_wins, &at_nine),
            ("after_the_draw", after_the_draw, &at_ones),
            ("next_encoder", next_encoder, &at_ones),
        ] {
            let (command_buffer, counts) = lane
                .grid(encode)
                .unwrap_or_else(|e| panic!("{sequence} on {name}: {e}"));
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(&counts, expected, "{sequence} on {name}: the counts");
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// An indirect strip of 90,000 positions restarting at 7 every 1000 of them, whose list (3 ·
/// 90 · 997 32-bit indices, 1,076,760 bytes) passes the 1 MiB every device's index pool held
/// previously, where its command buffer failed `IndexPoolOverflow`: it now counts what Metal's
/// answer drawn plainly counts, and its command buffer completes. The restarts at 7 show the
/// rewrite pass ran: a strip drawn without it would draw through vertex 7.
#[test]
fn a_restarted_list_past_the_old_index_pool_draws_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let positions = buffer_with(&lane.device, &scattered(false)).unwrap();
        let indices: Vec<u32> = (0..90_000u32)
            .map(|i| match i % 1000 {
                999 => 7,
                _ => ((i * 11 + 1) % SCATTERED) | 8,
            })
            .collect();
        let case = Chosen {
            name: "past_the_old_pool",
            primitive: PrimitiveType::TriangleStrip,
            index_type: IndexType::UInt32,
            count: indices.len() as u32,
            indices,
            k: 7,
            base_vertex: 0,
            form: Form::Indirect(0),
            culled: false,
            offset: 0,
        };
        let list_bytes: u64 = case
            .pieces()
            .iter()
            .map(|piece| (piece.len() as u64).saturating_sub(2) * 3 * 4)
            .sum();
        assert!(list_bytes > 1 << 20, "{list_bytes} bytes");
        let expected = lane.plainly(&case, &positions).unwrap();
        let (counts, _) = lane.restarted(&case, &positions).unwrap();
        assert_eq!(counts, expected, "{} on {name}: the counts", case.name);
        println!(
            "CBF-MARK {name}: an indirect strip restarting at 7, its list {list_bytes} bytes in the \
             index pool, completed with Metal's counts"
        );
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// A strip of a million positions, a restart every 1000 of them, counts what Metal's answer drawn
/// plainly counts; the test prints both command buffers' GPU times, and that of the strip drawn
/// at the all-ones index (no pass), whose difference is the index rewrite pass's cost (run it with
/// `--no-capture` to see them).
#[test]
fn a_million_position_strip_restarts_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let positions = buffer_with(&lane.device, &scattered(false)).unwrap();
        let indices: Vec<u32> = (0..1_000_000u32)
            .map(|i| match i % 1000 {
                999 => 7,
                _ => ((i * 11 + 1) % SCATTERED) | 8,
            })
            .collect();
        let case = |k| Chosen {
            name: "million",
            primitive: PrimitiveType::TriangleStrip,
            index_type: IndexType::UInt32,
            count: indices.len() as u32,
            indices: indices.clone(),
            k,
            base_vertex: 0,
            form: Form::Direct,
            culled: false,
            offset: 0,
        };
        let restarted = case(7);
        let expected = lane.plainly(&restarted, &positions).unwrap();
        // Warm the pass's kernel and the pipelines, then time.
        lane.restarted(&restarted, &positions).unwrap();
        let (counts, with_pass) = lane.restarted(&restarted, &positions).unwrap();
        assert_eq!(counts, expected, "on {name}: the counts");
        // The list the pass writes, drawn without it: the pass's cost is the difference.
        lane.listed(&restarted, &positions).unwrap();
        let (listed_counts, list_alone) = lane.listed(&restarted, &positions).unwrap();
        assert_eq!(listed_counts, expected, "on {name}: the list's counts");
        let at_all_ones = case(u64::from(u32::MAX));
        lane.restarted(&at_all_ones, &positions).unwrap();
        let (_, without_pass) = lane.restarted(&at_all_ones, &positions).unwrap();
        let pass = with_pass - list_alone;
        eprintln!(
            "restart pass on {name}: 1000000 positions, GPU {with_pass:.3} ms restarting at 7 \
             (the pass and the list), {list_alone:.3} ms for the list alone (so {pass:.3} ms for \
             the pass), {without_pass:.3} ms for the strip at the all-ones index (no pass)"
        );
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
