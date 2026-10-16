//! Device tests of `[[primitive_id]]` on the device the cfg selects, under the validation layer:
//! as the device has it, and with `geometryShader` and `tessellationShader` withheld, where the
//! fragment function reads the ID from a flat varying its pipeline's vertex function writes
//! (`LoweringInputs::primitive_id_varying`). The owned fixture `primitive_ids` places each vertex
//! at the pixel position buffer 0 (or attribute 0) holds for its vertex ID, a row stride lower per
//! instance, and `pi_ids` returns `(primitive_id, vertex_id, instance_id, 1)` of the primitive's
//! flat vertex into an `RGBA32Uint` attachment cleared to 0, so each covered pixel says which
//! primitive Metal would name there.
//!
//! Metal's answers (probes on the Apple M3 Max, macOS 27.0.1, the fixture's functions, 15 draws, 0
//! mismatches): `[[primitive_id]]` is the primitive's index among the draw's primitives of its
//! instance, counted from 0 in each instance; the vertex start, the base instance and the base
//! vertex do not move it; a flat varying takes the primitive's first vertex, so the primitive's
//! first vertex is at stream position `3i` of a triangle list's primitive `i`, `2i` of a line
//! list's, and `i` of a strip's or of points. [`model`] states that, and every check compares each
//! covered pixel with it and asks every primitive of every instance to cover a pixel.
//!
//! The varying's vertex function reads the draw's first vertex as `[[base_vertex]]` is read: the
//! `BaseVertex` builtin with `shaderDrawParameters`, so an indirect draw stays on the GPU, and the
//! push lane the binder writes without it, a third path, where an indirect draw's arguments
//! would be read on the CPU, which the runtime does not do: `NotImplemented(IndirectDrawArguments)`
//! there. On the paths without the features an indexed draw, whose primitives share vertices,
//! runs de-indexed (`VariantKey::primitive_id_indices`): its indices drawn as vertices from 0,
//! each read at its address. Its strip and a vertex function that writes memory are refused
//! (`NotImplemented(PrimitiveId)`).

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{BindError, Feature, Needs};
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage, VertexFormat};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandQueue, Device, DeviceOptions, Error, Library, LoadAction, PipelineOptions,
    Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap, StoreAction,
    TextureDescriptor,
};

/// The target's width and height, and the rows each instance moves down.
const WIDTH: u64 = 64;
const HEIGHT: u64 = 32;
const ROW: f32 = 4.0;

/// How the lane's device reads the ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// The device as the lane makes it (`PrimitiveId` through `Geometry` or `Tessellation`).
    Device,
    /// With both features withheld: the varying, its first vertex the `BaseVertex` builtin.
    Withheld,
    /// With `shaderDrawParameters` withheld too: the varying, its first vertex the binder's push
    /// lane.
    PushedBase,
}

const PATHS: [Path; 3] = [Path::Device, Path::Withheld, Path::PushedBase];

/// A draw of the probe's: its vertex function, primitive type, start, count, instances and base
/// instance, whether it is indirect, and its indices (with their base vertex) if indexed.
#[derive(Clone, Copy)]
struct Draw {
    what: &'static str,
    vertex: &'static str,
    primitive: PrimitiveType,
    start: u32,
    count: u32,
    instances: u32,
    base_instance: u32,
    indirect: bool,
    indexed: Option<(&'static [u16], i32)>,
}

const fn draw(what: &'static str, primitive: PrimitiveType, start: u32, count: u32) -> Draw {
    Draw {
        what,
        vertex: "pi_place",
        primitive,
        start,
        count,
        instances: 1,
        base_instance: 0,
        indirect: false,
        indexed: None,
    }
}

/// The probe's draws without indices.
const DRAWS: [Draw; 8] = [
    draw("triangles", PrimitiveType::Triangle, 0, 12),
    Draw {
        instances: 2,
        base_instance: 3,
        ..draw("triangles from 6", PrimitiveType::Triangle, 6, 12)
    },
    Draw {
        instances: 2,
        ..draw(
            "a triangle strip from 2",
            PrimitiveType::TriangleStrip,
            2,
            7,
        )
    },
    draw("lines from 4", PrimitiveType::Line, 4, 10),
    Draw {
        instances: 2,
        base_instance: 1,
        ..draw("a line strip from 1", PrimitiveType::LineStrip, 1, 6)
    },
    Draw {
        vertex: "pi_points",
        instances: 2,
        ..draw("points from 5", PrimitiveType::Point, 5, 6)
    },
    Draw {
        vertex: "pi_attrib",
        instances: 2,
        ..draw("attribute triangles from 3", PrimitiveType::Triangle, 3, 9)
    },
    Draw {
        instances: 2,
        base_instance: 2,
        indirect: true,
        ..draw("indirect triangles from 3", PrimitiveType::Triangle, 3, 9)
    },
];

/// The probe's indexed triangles: 12 indices from index 2, base vertex 10, two instances.
const PROBE_INDEXED: Draw = Draw {
    instances: 2,
    base_instance: 1,
    indexed: Some((&[9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 13], 10)),
    ..draw("indexed triangles", PrimitiveType::Triangle, 2, 12)
};

/// Indexed draws, each with whether the paths without the features refuse it: the probe's
/// triangles, lines and points, which the de-indexed form draws; a strip, whose restart index
/// the form does not take yet; and the probe's triangles through `pi_counts`, which counts its
/// invocations in memory and would count once per index.
const INDEXED: [(Draw, bool); 5] = [
    (PROBE_INDEXED, false),
    (
        Draw {
            instances: 2,
            indexed: Some((&[5, 4, 3, 2, 1, 0, 7, 6], 3)),
            ..draw("indexed lines", PrimitiveType::Line, 0, 8)
        },
        false,
    ),
    (
        Draw {
            vertex: "pi_points",
            instances: 2,
            base_instance: 4,
            indexed: Some((&[3, 1, 2, 0, 5, 4], 0)),
            ..draw("indexed points", PrimitiveType::Point, 0, 6)
        },
        false,
    ),
    (
        Draw {
            indexed: Some((&[0, 1, 2, 3, 4, 5, 6], 0)),
            ..draw(
                "an indexed triangle strip",
                PrimitiveType::TriangleStrip,
                0,
                7,
            )
        },
        true,
    ),
    (
        Draw {
            vertex: "pi_counts",
            ..PROBE_INDEXED
        },
        true,
    ),
];

fn per_primitive(primitive: PrimitiveType) -> u32 {
    match primitive {
        PrimitiveType::Triangle => 3,
        PrimitiveType::Line => 2,
        _ => 1,
    }
}

/// Each stream position's vertex ID.
fn stream(d: &Draw) -> Vec<u32> {
    (0..d.count)
        .map(|p| match d.indexed {
            Some((indices, base)) => {
                (i64::from(indices[(d.start + p) as usize]) + i64::from(base)) as u32
            }
            None => d.start + p,
        })
        .collect()
}

/// The pixel position of stream position `p` of a draw of `primitive` (the probe's layout).
fn layout(primitive: PrimitiveType, p: u32) -> [f32; 2] {
    let p32 = p as f32;
    match primitive {
        PrimitiveType::Point => [4.0 * p32 + 2.5, 1.5],
        PrimitiveType::Line => [
            4.0 * (p / 2) as f32 + if p % 2 == 1 { 3.5 } else { 0.5 },
            1.5,
        ],
        PrimitiveType::LineStrip => [3.0 * p32 + 1.5, 1.5],
        PrimitiveType::Triangle => {
            let x = 4.0 * (p / 3) as f32;
            match p % 3 {
                0 => [x + 0.25, 0.25],
                1 => [x + 3.75, 0.25],
                _ => [x + 0.25, 3.75],
            }
        }
        PrimitiveType::TriangleStrip => [2.0 * p32 + 0.25, if p % 2 == 1 { 3.75 } else { 0.25 }],
    }
}

/// Metal's `[[primitive_id]]` of the primitive whose first vertex is at stream position `p`, if a
/// primitive's first vertex is there.
fn model(d: &Draw, p: u32) -> Option<u32> {
    let n = per_primitive(d.primitive);
    let span = match d.primitive {
        PrimitiveType::TriangleStrip => 3,
        PrimitiveType::LineStrip => 2,
        _ => n,
    };
    let primitives = match n {
        1 => (d.count + 1).saturating_sub(span),
        _ => d.count / n,
    };
    (p.is_multiple_of(n) && p / n < primitives).then_some(p / n)
}

/// The lane's device, queue and the fixture's library.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn pipeline(&self, vertex: &str) -> Result<RenderPipelineState, Error> {
        let function = |name| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
        descriptor.fragment = Some(function("pi_ids"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        if vertex == "pi_attrib" {
            let mut v = VertexDescriptor::new();
            v.layouts
                .insert(0, VertexBufferLayoutDescriptor::new(Some(8)));
            v.attributes.insert(
                0,
                VertexAttributeDescriptor::new(VertexFormat::Float2, 0, 0),
            );
            descriptor.vertex_descriptor = Some(v);
        }
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    /// The target's texels after draw `d`, each its four words; or the draw call's refusal.
    fn run(&self, d: &Draw) -> Result<Vec<[u32; 4]>, Error> {
        let pipeline = self.pipeline(d.vertex)?;
        let mut xy = vec![-100.0f32; 2 * 64];
        for (p, &vid) in stream(d).iter().enumerate() {
            let [x, y] = layout(d.primitive, p as u32);
            xy[2 * vid as usize] = x;
            xy[2 * vid as usize + 1] = y;
        }
        let bytes: Vec<u8> = xy.iter().flat_map(|v| v.to_le_bytes()).collect();
        let vertices = buffer_with(&self.device, &bytes).unwrap();
        let frame: Vec<u8> = [WIDTH as f32, HEIGHT as f32, ROW, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA32Uint, WIDTH, HEIGHT, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&descriptor).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        color.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_vertex_buffer(Some(&vertices), 0, 0).unwrap();
        e.set_vertex_bytes(&frame, 1).unwrap();
        let counter = buffer_with(&self.device, &[0; 4]).unwrap();
        if d.vertex == "pi_counts" {
            e.set_vertex_buffer(Some(&counter), 0, 2).unwrap();
        }
        let drawn = match (d.indexed, d.indirect) {
            (None, false) => {
                e.draw_primitives(d.primitive, d.start, d.count, d.instances, d.base_instance)
            }
            (None, true) => {
                let args: Vec<u8> = [d.count, d.instances, d.start, d.base_instance]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                let arguments = buffer_with(&self.device, &args).unwrap();
                e.draw_primitives_indirect(d.primitive, &arguments, 0)
            }
            (Some((indices, base)), _) => {
                let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
                let index_buffer = buffer_with(&self.device, &bytes).unwrap();
                e.draw_indexed_primitives(
                    d.primitive,
                    d.count,
                    IndexType::UInt16,
                    &index_buffer,
                    u64::from(d.start) * 2,
                    d.instances,
                    base,
                    d.base_instance,
                )
            }
        };
        e.end_encoding().unwrap();
        drawn?;
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{}", d.what);
        let mut out = vec![0u8; (WIDTH * HEIGHT * 16) as usize];
        target
            .get_bytes(
                &mut out,
                WIDTH * 16,
                WIDTH * HEIGHT * 16,
                Region::new_2d(0, 0, WIDTH, HEIGHT),
                0,
                0,
            )
            .unwrap();
        Ok(out
            .as_chunks::<16>()
            .0
            .iter()
            .map(|t| {
                let (words, _) = t.as_chunks::<4>();
                std::array::from_fn(|i| u32::from_le_bytes(words[i]))
            })
            .collect())
    }

    /// Draw `d` and compare each covered pixel's ID with [`model`]'s, every primitive of every
    /// instance covering one.
    fn check(&self, d: &Draw) {
        let texels = self.run(d).unwrap_or_else(|e| panic!("{}: {e}", d.what));
        let positions: std::collections::BTreeMap<u32, u32> = stream(d)
            .iter()
            .enumerate()
            .map(|(p, &vid)| (vid, p as u32))
            .collect();
        let mut drawn = std::collections::BTreeSet::new();
        for t in texels.iter().filter(|t| t[3] == 1) {
            let [id, vid, iid, _] = *t;
            let p = positions[&vid];
            assert_eq!(
                Some(id),
                model(d, p),
                "{}: vertex {vid} (position {p}) of instance {iid}",
                d.what
            );
            assert!(
                (d.base_instance..d.base_instance + d.instances).contains(&iid),
                "{}: instance {iid}",
                d.what
            );
            drawn.insert((iid, id));
        }
        let primitives = (0..d.count).filter_map(|p| model(d, p)).count();
        assert_eq!(
            drawn.len(),
            primitives * d.instances as usize,
            "{}: primitives covering a pixel",
            d.what
        );
    }
}

/// `check` on the lane device of `path`; the validation layer must report nothing.
fn on_path(path: Path, check: impl FnOnce(&Lane)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        withhold(path, options)
    })
    .unwrap();
    let bitcode = llvm_assemble(&engine_dir().join("primitive_ids.ll")).unwrap();
    let lane = Lane {
        queue: device.new_command_queue().unwrap(),
        library: device.new_library_with_air(&bitcode).unwrap(),
        device,
    };
    check(&lane);
    drop(lane);
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

fn withhold(path: Path, options: &mut DeviceOptions) {
    if path != Path::Device {
        options.withheld_features.extend([
            VkFeature::GeometryShader,
            VkFeature::TessellationShader,
            VkFeature::ShaderTessellationAndGeometryPointSize,
            VkFeature::MultiviewGeometryShader,
            VkFeature::MultiviewTessellationShader,
        ]);
    }
    if path == Path::PushedBase {
        options
            .withheld_features
            .insert(VkFeature::ShaderDrawParameters);
    }
}

/// Every draw without indices names Metal's primitive at every pixel, through the device's
/// builtin and through the varying, whose first vertex is the `BaseVertex` builtin or the push
/// lane. On the push lane's path an indirect draw is refused for the CPU read of its arguments.
#[test]
fn every_draw_without_indices_names_metal_s_primitive() {
    for path in PATHS {
        on_path(path, |lane| {
            for d in &DRAWS {
                if path == Path::PushedBase && d.indirect {
                    match lane.run(d) {
                        Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments)) => {}
                        other => panic!("{}: {:?}", d.what, other.map(|_| ())),
                    }
                } else {
                    lane.check(d);
                }
            }
        });
    }
}

/// Whether `e` refuses the draw as not implemented for `[[primitive_id]]`, at bind or at the
/// variant's compile.
fn refuses_primitive_id(e: &Error) -> bool {
    match e {
        Error::Bind(BindError::Unsupported(u)) | Error::Unsupported(u) => {
            u.feature == Feature::PrimitiveId && u.needs == Needs::NotImplemented
        }
        _ => false,
    }
}

/// An indexed draw names Metal's primitive through the device's builtin and, de-indexed, through
/// the varying: triangle and line lists and points. The varying's strip and its vertex function
/// that writes memory are refused.
#[test]
fn an_indexed_draw_is_metal_s_or_refused() {
    for path in PATHS {
        on_path(path, |lane| {
            for (d, refused) in &INDEXED {
                if *refused && path != Path::Device {
                    match lane.run(d) {
                        Err(e) if refuses_primitive_id(&e) => {}
                        other => panic!("{path:?} {}: {:?}", d.what, other.map(|_| ())),
                    }
                } else {
                    lane.check(d);
                }
            }
        });
    }
}
