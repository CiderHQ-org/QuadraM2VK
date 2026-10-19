//! The shaders of the device's own checks of its rasterizer's answers (`device::checks`), which the
//! runtime draws on the device before it takes a fact a check decides: each module holds the
//! instructions translation writes where the profile takes the fact (a deviation) or leaves the
//! workaround out (a defect), so the draw shows whether the device answers them as Metal does.
//!
//! Every module numbers its resources in set [`RasterCheck::SET`]: the vertex records at
//! [`RasterCheck::VERTICES_BINDING`] (the vertex stage's), the words a draw or dispatch leaves at
//! [`RasterCheck::RESULTS_BINDING`], the parameters it reads at [`RasterCheck::PARAMS_BINDING`],
//! the image it reads or writes at [`RasterCheck::IMAGE_BINDING`] (the fragment or compute
//! stage's), and the cube gradients checks' four combined image samplers from
//! [`RasterCheck::CUBE_ANISOTROPIC_BINDING`]. Pixel `(x, y)` of a target `width` wide is pixel `y · width + x`, and its sample `s` is
//! slot `4 · pixel + s` whatever the draw's sample count ([`RasterCheck::SAMPLE_SLOTS`]).
//!
//! - **The vertex stage** ([`raster_check_vertex`]): vertex `v` of instance `k` reads the records
//!   `2 (3k + v)` (its clip position) and `2 (3k + v) + 1` (its attribute) of the `vec4` buffer, so a
//!   draw of 3 vertices with first instance `k` draws triangle `k`. It writes the position, the
//!   attribute to the varyings at locations [`RasterCheck::ATTRIBUTE_LOCATION`] and
//!   [`RasterCheck::ATTRIBUTE_NO_PERSPECTIVE_LOCATION`], and the window depth `z / w` to
//!   [`RasterCheck::WINDOW_DEPTH_LOCATION`], as a vertex function compiled for a computed sample
//!   depth writes it (`lower::stage_io`'s window depth).
//! - **The fragment's depth** ([`fragcoord_depth_check_fragment`], `CenterFragCoordDepth` and
//!   `SampleFragCoordDepth`): a per-sample invocation (it reads `SampleId`) stores two words at slot
//!   `2 (4 · pixel + s)`: `FragCoord.z`, and the sample's depth computed as a function testing its
//!   own depth computes it on a device whose `FragCoord.z` is the pixel centre's
//!   (`lower::fragment::depth_test`'s `sample_depth`): `(v(s) · (far − near) + near) + (FragCoord.z
//!   − (v · (far − near) + near))`, `v(s)` the window depth through `InterpolateAtSample`, `v` read
//!   as it is, each operation rounded on its own, `near` and `far` the parameters' first two words.
//!   [`depth_samples_check_kernel`] then stores the depth the device's own test stored at each
//!   sample at slot `4 · pixel + s`.
//! - **The centre varyings** ([`center_varyings_check_fragment`], `MetalCenterVaryings`): a
//!   per-sample invocation stores the attribute's `x` and `y` read as they are, smooth and without
//!   perspective, four words at slot `4 (4 · pixel + s)`: the read translation writes for a center
//!   varying where the device reads it at the pixel centre.
//! - **The coverage counts** ([`coverage_count_check_fragment`],
//!   `MemoryWritingFragmentsRunPerSample` and `ClippedTrianglesSplitPixelFragments`): an invocation
//!   that reads `SampleMask` and writes memory, without sample shading, adds 1 to word `3 · pixel`,
//!   ORs its mask into word `3 · pixel + 1` and ANDs it into word `3 · pixel + 2`, as a fragment
//!   function that reads `[[sample_mask]]` and writes memory runs where the device runs it once per
//!   covered sample (`lower::fragment::coverage`'s `CoverageLoop::DriverRunsPerSample`).
//! - **The early tests** ([`early_tests_check_fragment`], `EarlyTestsDropMultisampledImageWrites`):
//!   a function declaring early fragment tests stores 1 at word `pixel` and at texel `(x, y)` of an
//!   `R32ui` storage image, as the per-sample storage form of a framebuffer fetch writes under
//!   early tests.
//! - **The barycentric pair** ([`barycentric_pair_check_fragment`], `BothBarycentricFormsRejected`):
//!   a fragment that reads `BaryCoordKHR` and `BaryCoordNoPerspKHR`, as a fragment function that
//!   declares both `[[barycentric_coord]]` forms reads them, and writes their sum to the color at
//!   location 0; its control reads `BaryCoordKHR` alone. A pipeline of each is made, never drawn.
//! - **The cube gradients** ([`cube_gradients_check_fragment`] and [`cube_gradients_check_kernel`],
//!   `GraphicsCubeGradientRoute` and `GraphicsCubeGradientTangents`): lane `l` (the fragment's pixel
//!   `x` in a target one pixel high, the kernel's invocation `x`) reads its direction `p`, `dPdx`
//!   and `dPdy` from the parameters' `vec4` records `3l` to `3l + 2`, and stores samples of the cube
//!   at `p` and of the cube array's layer [`RasterCheck::CUBE_ARRAY_LAYER`] at `p`, four words each,
//!   from word `4 · 8 l` (the fragment) or `4 · 4 l` (the kernel). The kernel hands the gradients as
//!   given (`CubeFootprint::Metal`), through the anisotropic sampler, then the isotropic one. The
//!   fragment hands them in the form each graphics route takes (`lower::sampler`'s
//!   `cube_footprint`): macOS 26.7's anisotropic form through the anisotropic sampler, its
//!   isotropic form through the isotropic one, then the projected form of macOS 27 through each.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::lower::sampler::CubeFootprint;
use crate::pipeline::sym::{
    CombinedSampler, ResourceKind, ResourceSlot, SamplerRef, Stage, Symbol, VaryingKey,
};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, GlslOp,
    ImageOperands, ImageType, ImageUsage, Interpolation, IoDecor, IoDir, LayoutRule, MemAccess,
    Scope, Semantics, StorageClass, TexelMemory, Type, TypedModule,
};

/// One module of a raster check, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RasterCheck {
    /// SPIR-V 1.5 words, entry point `main`, resources numbered as the module docs state.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within its profile).
    pub requirements: Requirements,
}

impl RasterCheck {
    /// The descriptor set of every resource.
    pub const SET: u32 = 0;
    /// The vertex records, `vec4`s, a storage buffer the vertex stage reads.
    pub const VERTICES_BINDING: u32 = 0;
    /// The results, 4-byte words, a storage buffer the fragment or compute stage writes.
    pub const RESULTS_BINDING: u32 = 1;
    /// The parameters, a storage buffer the fragment stage reads: the viewport's depth range,
    /// `near` then `far` (`float`s), or the words the depth writes check writes as its depths
    /// ([`depth_writes_check_fragment`](crate::bind::depth_writes_check_fragment)).
    pub const PARAMS_BINDING: u32 = 2;
    /// The image: the early tests' `R32ui` storage image, or the stored depth's multisampled
    /// sampled image.
    pub const IMAGE_BINDING: u32 = 3;
    /// The records of one vertex: its clip position, then its attribute.
    pub const RECORDS_PER_VERTEX: u32 = 2;
    /// The location of the attribute's smooth varying.
    pub const ATTRIBUTE_LOCATION: u32 = 0;
    /// The location of the attribute's no-perspective varying.
    pub const ATTRIBUTE_NO_PERSPECTIVE_LOCATION: u32 = 1;
    /// The location of the window depth's varying.
    pub const WINDOW_DEPTH_LOCATION: u32 = 2;
    /// The slots of a pixel's samples: 4 whatever a draw's sample count.
    pub const SAMPLE_SLOTS: u32 = 4;
    /// The stored depths kernel's workgroup, along x and y; its z is the sample.
    pub const DEPTH_SAMPLES_LOCAL_SIZE: u32 = 8;
    /// The cube gradients checks' cube through the anisotropic sampler, a combined image sampler.
    pub const CUBE_ANISOTROPIC_BINDING: u32 = 4;
    /// The cube through the isotropic sampler.
    pub const CUBE_ISOTROPIC_BINDING: u32 = 5;
    /// The cube array through the anisotropic sampler.
    pub const CUBE_ARRAY_ANISOTROPIC_BINDING: u32 = 6;
    /// The cube array through the isotropic sampler.
    pub const CUBE_ARRAY_ISOTROPIC_BINDING: u32 = 7;
    /// The `vec4` records of one cube lane in the parameters: its direction, `dPdx` and `dPdy`.
    pub const CUBE_LANE_RECORDS: u32 = 3;
    /// The cube array's layer the checks sample.
    pub const CUBE_ARRAY_LAYER: u32 = 1;
    /// The samples a lane of the cube gradients fragment stage stores, four words each.
    pub const CUBE_FRAGMENT_SAMPLES: u32 = 8;
    /// The samples a lane of the cube gradients kernel stores, four words each.
    pub const CUBE_KERNEL_SAMPLES: u32 = 4;
    /// The cube gradients kernel's workgroup, along x: a lane an invocation.
    pub const CUBE_KERNEL_LOCAL_SIZE: u32 = 8;
}

const VERTICES: ResourceSlot = ResourceSlot {
    stage: Stage::Vertex,
    kind: ResourceKind::Buffer,
    index: 0,
};
const RESULTS: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Buffer,
    index: 1,
};
const PARAMS: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Buffer,
    index: 2,
};
const IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Texture,
    index: 3,
};

/// The combined image sampler of the cube gradients checks at slot `index`: the texture of that
/// slot through the runtime sampler of the same slot.
const fn cube_slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::CombinedTextureSampler(CombinedSampler {
            sampler: SamplerRef::Slot(index),
            compare: false,
            integer: false,
            cube_gradient: false,
            raw: false,
        }),
        index,
    }
}

const CUBE_ANISOTROPIC: ResourceSlot = cube_slot(4);
const CUBE_ISOTROPIC: ResourceSlot = cube_slot(5);
const CUBE_ARRAY_ANISOTROPIC: ResourceSlot = cube_slot(6);
const CUBE_ARRAY_ISOTROPIC: ResourceSlot = cube_slot(7);

fn attribute() -> VaryingKey {
    VaryingKey::user(b"raster-check-attribute")
}

fn attribute_no_perspective() -> VaryingKey {
    VaryingKey::user(b"raster-check-attribute-no-perspective")
}

fn window_depth() -> VaryingKey {
    VaryingKey::user(b"raster-check-window-depth")
}

/// The fixed numbers of the modules' symbols.
fn number(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(RasterCheck::SET),
        Symbol::Binding(s) if *s == VERTICES => Some(RasterCheck::VERTICES_BINDING),
        Symbol::Binding(s) if *s == RESULTS => Some(RasterCheck::RESULTS_BINDING),
        Symbol::Binding(s) if *s == PARAMS => Some(RasterCheck::PARAMS_BINDING),
        Symbol::Binding(s) if *s == IMAGE => Some(RasterCheck::IMAGE_BINDING),
        Symbol::Binding(s) if *s == CUBE_ANISOTROPIC => Some(RasterCheck::CUBE_ANISOTROPIC_BINDING),
        Symbol::Binding(s) if *s == CUBE_ISOTROPIC => Some(RasterCheck::CUBE_ISOTROPIC_BINDING),
        Symbol::Binding(s) if *s == CUBE_ARRAY_ANISOTROPIC => {
            Some(RasterCheck::CUBE_ARRAY_ANISOTROPIC_BINDING)
        }
        Symbol::Binding(s) if *s == CUBE_ARRAY_ISOTROPIC => {
            Some(RasterCheck::CUBE_ARRAY_ISOTROPIC_BINDING)
        }
        Symbol::Location(k) if *k == attribute() => Some(RasterCheck::ATTRIBUTE_LOCATION),
        Symbol::Location(k) if *k == attribute_no_perspective() => {
            Some(RasterCheck::ATTRIBUTE_NO_PERSPECTIVE_LOCATION)
        }
        Symbol::Location(k) if *k == window_depth() => Some(RasterCheck::WINDOW_DEPTH_LOCATION),
        _ => None,
    }
}

/// The scalar and vector types the modules use.
struct Types {
    void: TypeId,
    u32_: TypeId,
    i32_: TypeId,
    f32_: TypeId,
    uvec2: TypeId,
    uvec3: TypeId,
    uvec4: TypeId,
    vec3: TypeId,
    vec4: TypeId,
}

impl Types {
    fn new(b: &mut ModuleBuilder) -> BResult<Types> {
        let u32_ = b.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let i32_ = b.ty(Type::Int {
            width: 32,
            signed: true,
        })?;
        let f32_ = b.ty(Type::Float { width: 32 })?;
        Ok(Types {
            void: b.ty(Type::Void)?,
            u32_,
            i32_,
            f32_,
            uvec2: b.ty(Type::Vector { elem: u32_, len: 2 })?,
            uvec3: b.ty(Type::Vector { elem: u32_, len: 3 })?,
            uvec4: b.ty(Type::Vector { elem: u32_, len: 4 })?,
            vec3: b.ty(Type::Vector { elem: f32_, len: 3 })?,
            vec4: b.ty(Type::Vector { elem: f32_, len: 4 })?,
        })
    }
}

/// A `Block { elem v[]; }` storage buffer bound at `sym`, `elem` of `stride` bytes.
fn buffer(
    b: &mut ModuleBuilder,
    sym: ResourceSlot,
    elem: TypeId,
    stride: u32,
) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(stride),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// A pointer to element `at` of the storage buffer `buffer`, or to its first element where `at`
/// lies past the buffer's end. A fragment stage's helper invocations shade the uncovered pixels of
/// a quad (or of a larger block), whose index may lie past a buffer sized to the target, and an
/// access past a buffer is undefined on a device without robust buffer access; a helper's stores
/// and atomics have no effect, so the bound changes no word a check reads.
fn element(f: &mut FunctionBuilder<'_>, buffer: GlobalId, at: Value) -> BResult<Value> {
    let zero = u32_value(f, 0)?;
    let buffer = f.module().global_value(buffer)?;
    let len = f.array_length(buffer, 0)?;
    let inside = f.binary(BinOp::ULessThan, at, len)?;
    let at = f.select(inside, at, zero)?;
    f.access_chain(buffer, &[zero, at])
}

/// Store `value` (a 32-bit scalar) as word `at` of the results.
fn store_word(
    f: &mut FunctionBuilder<'_>,
    t: &Types,
    results: GlobalId,
    at: Value,
    value: Value,
) -> BResult<()> {
    let bits = match f.value_ty(value)? == t.u32_ {
        true => value,
        false => f.convert(ConvOp::Bitcast, value, t.u32_)?,
    };
    let to = element(f, results, at)?;
    f.store(to, bits, MemAccess::default())
}

/// The loaded value of the input builtin `builtin`.
fn load_builtin(f: &mut FunctionBuilder<'_>, builtin: GlobalId) -> BResult<Value> {
    let ptr = f.module().global_value(builtin)?;
    f.load(ptr, MemAccess::default())
}

/// The pixel the invocation shades in a target `width` wide, `y · width + x`, from its
/// `FragCoord`.
fn pixel(f: &mut FunctionBuilder<'_>, t: &Types, coord: Value, width: u32) -> BResult<Value> {
    let x = f.composite_extract(coord, &[0])?;
    let y = f.composite_extract(coord, &[1])?;
    let x = f.convert(ConvOp::ConvertFToU, x, t.u32_)?;
    let y = f.convert(ConvOp::ConvertFToU, y, t.u32_)?;
    let width = u32_value(f, width)?;
    let row = f.binary(BinOp::IMul, y, width)?;
    f.binary(BinOp::IAdd, row, x)
}

/// The slot `4 · pixel + sample` of the invocation's sample, `sample` its `SampleId`.
fn sample_slot(
    f: &mut FunctionBuilder<'_>,
    t: &Types,
    pixel: Value,
    sample: Value,
) -> BResult<Value> {
    let slots = u32_value(f, RasterCheck::SAMPLE_SLOTS)?;
    let sample = f.convert(ConvOp::Bitcast, sample, t.u32_)?;
    let at = f.binary(BinOp::IMul, pixel, slots)?;
    f.binary(BinOp::IAdd, at, sample)
}

/// The shared vertex stage as a typed module (the module docs).
fn vertex_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let records = buffer(&mut b, VERTICES, t.vec4, 16)?;
    let vertex = b.builtin(crate::spv::BuiltIn::VertexIndex, IoDir::Input, t.i32_)?;
    let instance = b.builtin(crate::spv::BuiltIn::InstanceIndex, IoDir::Input, t.i32_)?;
    let position = b.builtin(crate::spv::BuiltIn::Position, IoDir::Output, t.vec4)?;
    let smooth = b.io(attribute(), IoDir::Output, t.vec4, IoDecor::default())?;
    let flat_free = b.io(
        attribute_no_perspective(),
        IoDir::Output,
        t.vec4,
        IoDecor::default(),
    )?;
    let depth = b.io(window_depth(), IoDir::Output, t.f32_, IoDecor::default())?;
    let mut f = b.function(t.void)?;
    let vertex = load_builtin(&mut f, vertex)?;
    let instance = load_builtin(&mut f, instance)?;
    let vertex = f.convert(ConvOp::Bitcast, vertex, t.u32_)?;
    let instance = f.convert(ConvOp::Bitcast, instance, t.u32_)?;
    let three = u32_value(&mut f, 3)?;
    let per_vertex = u32_value(&mut f, RasterCheck::RECORDS_PER_VERTEX)?;
    let one = u32_value(&mut f, 1)?;
    let first = f.binary(BinOp::IMul, instance, three)?;
    let index = f.binary(BinOp::IAdd, first, vertex)?;
    let at = f.binary(BinOp::IMul, index, per_vertex)?;
    let from = element(&mut f, records, at)?;
    let clip = f.load(from, MemAccess::default())?;
    let at = f.binary(BinOp::IAdd, at, one)?;
    let from = element(&mut f, records, at)?;
    let value = f.load(from, MemAccess::default())?;
    for (out, v) in [(position, clip), (smooth, value), (flat_free, value)] {
        let out = f.module().global_value(out)?;
        f.store(out, v, MemAccess::default())?;
    }
    // The window depth as a vertex function compiled for a computed sample depth stores it: the
    // division, unfused (`lower::stage_io`'s `store_window_depth`).
    let z = f.composite_extract(clip, &[2])?;
    let w = f.composite_extract(clip, &[3])?;
    let window = f.binary(BinOp::FDiv, z, w)?;
    let depth = f.module().global_value(depth)?;
    f.store(depth, window, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Vertex, id, Vec::new())?;
    b.finalize()
}

/// The fragment's depth check as a typed module, for a target `width` wide (the module docs).
fn fragcoord_depth_typed_module(width: u32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let results = buffer(&mut b, RESULTS, t.u32_, 4)?;
    let params = buffer(&mut b, PARAMS, t.f32_, 4)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, t.vec4)?;
    let sample = b.builtin(crate::spv::BuiltIn::SampleId, IoDir::Input, t.i32_)?;
    let window = b.io(
        window_depth(),
        IoDir::Input,
        t.f32_,
        IoDecor {
            interpolation: Interpolation::NoPerspective,
            ..IoDecor::default()
        },
    )?;
    let mut f = b.function(t.void)?;
    let coord = load_builtin(&mut f, coord)?;
    let sample = load_builtin(&mut f, sample)?;
    let pixel = pixel(&mut f, &t, coord, width)?;
    let slot = sample_slot(&mut f, &t, pixel, sample)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let near = element(&mut f, params, zero)?;
    let near = f.load(near, MemAccess::default())?;
    let far = element(&mut f, params, one)?;
    let far = f.load(far, MemAccess::default())?;
    let centre = f.composite_extract(coord, &[2])?;
    // `lower::fragment::depth_test`'s `sample_depth`, operation for operation.
    let var = f.module().global_value(window)?;
    let at_sample = f.glsl(GlslOp::InterpolateAtSample, &[var, sample])?;
    let at_centre = f.load(var, MemAccess::default())?;
    let range = f.binary_no_contraction(BinOp::FSub, far, near)?;
    let window_depth = |f: &mut FunctionBuilder<'_>, v: Value| -> BResult<Value> {
        let scaled = f.binary_no_contraction(BinOp::FMul, v, range)?;
        f.binary_no_contraction(BinOp::FAdd, scaled, near)
    };
    let at_sample = window_depth(&mut f, at_sample)?;
    let at_centre = window_depth(&mut f, at_centre)?;
    let carried = f.binary_no_contraction(BinOp::FSub, centre, at_centre)?;
    let computed = f.binary_no_contraction(BinOp::FAdd, at_sample, carried)?;
    let two = u32_value(&mut f, 2)?;
    let at = f.binary(BinOp::IMul, slot, two)?;
    store_word(&mut f, &t, results, at, centre)?;
    let at = f.binary(BinOp::IAdd, at, one)?;
    store_word(&mut f, &t, results, at, computed)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The stored depths kernel as a typed module, for a target `width` wide: invocation `(x, y, s)`
/// stores sample `s` of texel `(x, y)` of the multisampled depth at slot `4 (y · width + x) + s`.
/// A dispatch covers the target exactly: `width / 8` by `height / 8` by the sample count.
fn depth_samples_typed_module(width: u32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let results = buffer(&mut b, RESULTS, t.u32_, 4)?;
    let depth_ty = b.ty(Type::Image(ImageType {
        sampled_type: t.f32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: true,
        usage: ImageUsage::Sampled,
        format: crate::spv::ImageFormat::Unknown,
    }))?;
    let depth = b.resource(IMAGE, StorageClass::UniformConstant, depth_ty)?;
    let invocation = b.builtin(
        crate::spv::BuiltIn::GlobalInvocationId,
        IoDir::Input,
        t.uvec3,
    )?;
    let mut f = b.function(t.void)?;
    let invocation = load_builtin(&mut f, invocation)?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let s = f.composite_extract(invocation, &[2])?;
    let texel = f.composite_construct(t.uvec2, &[x, y])?;
    let depth = load_builtin(&mut f, depth)?;
    let ops = ImageOperands {
        sample: Some(s),
        ..ImageOperands::default()
    };
    let stored = f.fetch(depth, texel, &ops)?;
    let stored = f.composite_extract(stored, &[0])?;
    let width = u32_value(&mut f, width)?;
    let slots = u32_value(&mut f, RasterCheck::SAMPLE_SLOTS)?;
    let row = f.binary(BinOp::IMul, y, width)?;
    let pixel = f.binary(BinOp::IAdd, row, x)?;
    let at = f.binary(BinOp::IMul, pixel, slots)?;
    let at = f.binary(BinOp::IAdd, at, s)?;
    store_word(&mut f, &t, results, at, stored)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: RasterCheck::DEPTH_SAMPLES_LOCAL_SIZE,
            y: RasterCheck::DEPTH_SAMPLES_LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The centre varyings check as a typed module, for a target `width` wide (the module docs).
fn center_varyings_typed_module(width: u32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let results = buffer(&mut b, RESULTS, t.u32_, 4)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, t.vec4)?;
    let sample = b.builtin(crate::spv::BuiltIn::SampleId, IoDir::Input, t.i32_)?;
    let smooth = b.io(attribute(), IoDir::Input, t.vec4, IoDecor::default())?;
    let linear = b.io(
        attribute_no_perspective(),
        IoDir::Input,
        t.vec4,
        IoDecor {
            interpolation: Interpolation::NoPerspective,
            ..IoDecor::default()
        },
    )?;
    let mut f = b.function(t.void)?;
    let coord = load_builtin(&mut f, coord)?;
    let sample = load_builtin(&mut f, sample)?;
    let pixel = pixel(&mut f, &t, coord, width)?;
    let slot = sample_slot(&mut f, &t, pixel, sample)?;
    let four = u32_value(&mut f, 4)?;
    let one = u32_value(&mut f, 1)?;
    let mut at = f.binary(BinOp::IMul, slot, four)?;
    for input in [smooth, linear] {
        let value = load_builtin(&mut f, input)?;
        for lane in 0..2 {
            let v = f.composite_extract(value, &[lane])?;
            store_word(&mut f, &t, results, at, v)?;
            at = f.binary(BinOp::IAdd, at, one)?;
        }
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The coverage counts check as a typed module, for a target `width` wide (the module docs).
fn coverage_count_typed_module(width: u32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let results = buffer(&mut b, RESULTS, t.u32_, 4)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, t.vec4)?;
    let one_mask = b.constant(Constant::Scalar {
        ty: t.u32_,
        bits: ConstBits(1),
    })?;
    let masks = b.ty(Type::Array {
        elem: t.i32_,
        len: one_mask,
        stride: None,
    })?;
    let mask = b.builtin(crate::spv::BuiltIn::SampleMask, IoDir::Input, masks)?;
    let mut f = b.function(t.void)?;
    let coord = load_builtin(&mut f, coord)?;
    let pixel = pixel(&mut f, &t, coord, width)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let three = u32_value(&mut f, 3)?;
    let mask = f.module().global_value(mask)?;
    let mask = f.access_chain(mask, &[zero])?;
    let mask = f.load(mask, MemAccess::default())?;
    let mask = f.convert(ConvOp::Bitcast, mask, t.u32_)?;
    let mut at = f.binary(BinOp::IMul, pixel, three)?;
    for (op, value) in [
        (AtomicOp::IAdd, one),
        (AtomicOp::Or, mask),
        (AtomicOp::And, mask),
    ] {
        let to = element(&mut f, results, at)?;
        f.atomic(op, to, Scope::Device, Semantics::RELAXED, value)?;
        at = f.binary(BinOp::IAdd, at, one)?;
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The early tests check as a typed module, for a target `width` wide (the module docs).
fn early_tests_typed_module(width: u32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let results = buffer(&mut b, RESULTS, t.u32_, 4)?;
    let image_ty = b.ty(Type::Image(ImageType {
        sampled_type: t.u32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Storage,
        format: crate::spv::ImageFormat::R32ui,
    }))?;
    let image = b.resource(IMAGE, StorageClass::UniformConstant, image_ty)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, t.vec4)?;
    let mut f = b.function(t.void)?;
    let coord = load_builtin(&mut f, coord)?;
    let at = pixel(&mut f, &t, coord, width)?;
    let one = u32_value(&mut f, 1)?;
    let zero = u32_value(&mut f, 0)?;
    store_word(&mut f, &t, results, at, one)?;
    let x = f.composite_extract(coord, &[0])?;
    let y = f.composite_extract(coord, &[1])?;
    let x = f.convert(ConvOp::ConvertFToU, x, t.u32_)?;
    let y = f.convert(ConvOp::ConvertFToU, y, t.u32_)?;
    let texel = f.composite_construct(t.uvec2, &[x, y])?;
    let value = f.composite_construct(t.uvec4, &[one, zero, zero, zero])?;
    let image = load_builtin(&mut f, image)?;
    f.write(
        image,
        texel,
        value,
        &ImageOperands::default(),
        TexelMemory::default(),
    )?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::Fragment,
        id,
        vec![ExecMode::OriginUpperLeft, ExecMode::EarlyFragmentTests],
    )?;
    b.finalize()
}

/// The barycentric pair check as a typed module (the module docs), or its control where not
/// `both`.
fn barycentric_pair_typed_module(both: bool) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let perspective = b.builtin(crate::spv::BuiltIn::BaryCoordKHR, IoDir::Input, t.vec3)?;
    let linear = match both {
        true => Some(b.builtin(
            crate::spv::BuiltIn::BaryCoordNoPerspKHR,
            IoDir::Input,
            t.vec3,
        )?),
        false => None,
    };
    let color = b.fixed_io(0, IoDir::Output, t.vec4, false)?;
    let mut f = b.function(t.void)?;
    let mut sum = load_builtin(&mut f, perspective)?;
    if let Some(linear) = linear {
        let linear = load_builtin(&mut f, linear)?;
        sum = f.binary(BinOp::FAdd, sum, linear)?;
    }
    let one = f.module().constant(Constant::Scalar {
        ty: t.f32_,
        bits: ConstBits(u64::from(1f32.to_bits())),
    })?;
    let one = f.module().const_value(one)?;
    let parts: Vec<Value> = (0..3)
        .map(|lane| f.composite_extract(sum, &[lane]))
        .collect::<BResult<_>>()?;
    let value = f.composite_construct(t.vec4, &[parts[0], parts[1], parts[2], one])?;
    let color = f.module().global_value(color)?;
    f.store(color, value, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The combined image samplers of the cube gradients checks.
struct CubeImages {
    cube_anisotropic: GlobalId,
    cube_isotropic: GlobalId,
    array_anisotropic: GlobalId,
    array_isotropic: GlobalId,
}

impl CubeImages {
    fn new(b: &mut ModuleBuilder, t: &Types) -> BResult<CubeImages> {
        let sampled = |b: &mut ModuleBuilder, arrayed| -> BResult<TypeId> {
            let image = b.ty(Type::Image(ImageType {
                sampled_type: t.f32_,
                dim: Dim::Cube,
                depth: false,
                arrayed,
                multisampled: false,
                usage: ImageUsage::Sampled,
                format: crate::spv::ImageFormat::Unknown,
            }))?;
            b.ty(Type::SampledImage { image })
        };
        let cube = sampled(b, false)?;
        let array = sampled(b, true)?;
        let class = StorageClass::UniformConstant;
        Ok(CubeImages {
            cube_anisotropic: b.resource(CUBE_ANISOTROPIC, class, cube)?,
            cube_isotropic: b.resource(CUBE_ISOTROPIC, class, cube)?,
            array_anisotropic: b.resource(CUBE_ARRAY_ANISOTROPIC, class, array)?,
            array_isotropic: b.resource(CUBE_ARRAY_ISOTROPIC, class, array)?,
        })
    }

    /// The cube's and the cube array's through the anisotropic sampler, or the isotropic one.
    fn through(&self, anisotropic: bool) -> [(GlobalId, bool); 2] {
        match anisotropic {
            true => [
                (self.cube_anisotropic, false),
                (self.array_anisotropic, true),
            ],
            false => [(self.cube_isotropic, false), (self.array_isotropic, true)],
        }
    }
}

/// The cube gradients checks as a typed module: the fragment stage where `kernel` is false, else
/// the kernel (the module docs).
fn cube_gradients_typed_module(kernel: bool) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let results = buffer(&mut b, RESULTS, t.u32_, 4)?;
    let lanes = buffer(&mut b, PARAMS, t.vec4, 16)?;
    let images = CubeImages::new(&mut b, &t)?;
    let (place, forms): (GlobalId, &[(CubeFootprint, bool)]) = match kernel {
        true => (
            b.builtin(
                crate::spv::BuiltIn::GlobalInvocationId,
                IoDir::Input,
                t.uvec3,
            )?,
            &[(CubeFootprint::Metal, true), (CubeFootprint::Metal, false)],
        ),
        false => (
            b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, t.vec4)?,
            &[
                (CubeFootprint::AnisotropicSwizzled, true),
                (CubeFootprint::Isotropic, false),
                (CubeFootprint::Projected, true),
                (CubeFootprint::Projected, false),
            ],
        ),
    };
    let mut f = b.function(t.void)?;
    let place = load_builtin(&mut f, place)?;
    let lane = f.composite_extract(place, &[0])?;
    let lane = match kernel {
        true => lane,
        false => f.convert(ConvOp::ConvertFToU, lane, t.u32_)?,
    };
    let records = u32_value(&mut f, RasterCheck::CUBE_LANE_RECORDS)?;
    let one = u32_value(&mut f, 1)?;
    let mut at = f.binary(BinOp::IMul, lane, records)?;
    let mut read = Vec::new();
    for k in 0..RasterCheck::CUBE_LANE_RECORDS {
        if k > 0 {
            at = f.binary(BinOp::IAdd, at, one)?;
        }
        let from = element(&mut f, lanes, at)?;
        let record = f.load(from, MemAccess::default())?;
        read.push(f.vector_shuffle(record, record, &[0, 1, 2])?);
    }
    let (p, grads) = (read[0], (read[1], read[2]));
    let layer = f.module().constant(Constant::Scalar {
        ty: t.f32_,
        bits: ConstBits(u64::from((RasterCheck::CUBE_ARRAY_LAYER as f32).to_bits())),
    })?;
    let layer = f.module().const_value(layer)?;
    let [px, py, pz] = [0, 1, 2].map(|c| f.composite_extract(p, &[c]));
    let in_array = f.composite_construct(t.vec4, &[px?, py?, pz?, layer])?;
    let samples = match kernel {
        true => RasterCheck::CUBE_KERNEL_SAMPLES,
        false => RasterCheck::CUBE_FRAGMENT_SAMPLES,
    };
    let words = u32_value(&mut f, 4 * samples)?;
    let mut at = f.binary(BinOp::IMul, lane, words)?;
    for &(form, anisotropic) in forms {
        let (dx, dy) = crate::lower::sampler::cube_footprint(&mut f, p, grads, form)?;
        for (image, arrayed) in images.through(anisotropic) {
            let image = load_builtin(&mut f, image)?;
            let coord = if arrayed { in_array } else { p };
            let ops = ImageOperands {
                grad: Some((dx, dy)),
                ..ImageOperands::default()
            };
            let sample = f.sample(image, coord, None, false, &ops)?;
            for c in 0..4 {
                let v = f.composite_extract(sample, &[c])?;
                store_word(&mut f, &t, results, at, v)?;
                at = f.binary(BinOp::IAdd, at, one)?;
            }
        }
    }
    f.ret(None)?;
    let id = f.finish()?;
    match kernel {
        true => b.entry(
            ExecModel::GLCompute,
            id,
            vec![ExecMode::LocalSize {
                x: RasterCheck::CUBE_KERNEL_LOCAL_SIZE,
                y: 1,
                z: 1,
            }],
        )?,
        false => b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?,
    };
    b.finalize()
}

/// `m` verified against `device`'s profile (requirements included), serialized, and numbered
/// with the fixed numbers; `what` names it in the error where the device lacks what it needs.
fn finish(device: &Device, m: TypedModule, what: &str) -> Result<RasterCheck, Error> {
    let profile = device.profile();
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    #[cfg(test)]
    crate::bind::device_checks::kernel::record(&m);
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the {what} needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &number)?;
    Ok(RasterCheck {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

/// The raster checks' shared vertex stage of `device` (the module docs): built with the typed
/// builder, checked by the owned verifier against the device's profile, serialized, and numbered
/// with the fixed numbers of [`RasterCheck`].
pub fn raster_check_vertex(device: &Device) -> Result<RasterCheck, Error> {
    finish(
        device,
        vertex_typed_module()?,
        "raster check's vertex stage",
    )
}

/// The fragment's depth check's fragment stage of `device`, for a target `width` wide (the module
/// docs), built as [`raster_check_vertex`] is. An internal error where the device lacks
/// `sampleRateShading` or a fragment stage's storage writes (`fragmentStoresAndAtomics`), or, on a
/// portability-subset device, `shaderSampleRateInterpolationFunctions`.
pub fn fragcoord_depth_check_fragment(device: &Device, width: u32) -> Result<RasterCheck, Error> {
    finish(
        device,
        fragcoord_depth_typed_module(width)?,
        "fragment depth check",
    )
}

/// The stored depths kernel of `device`, for a target `width` wide (the module docs), built as
/// [`raster_check_vertex`] is.
pub fn depth_samples_check_kernel(device: &Device, width: u32) -> Result<RasterCheck, Error> {
    finish(
        device,
        depth_samples_typed_module(width)?,
        "stored depths kernel",
    )
}

/// The centre varyings check's fragment stage of `device`, for a target `width` wide (the module
/// docs), built as [`raster_check_vertex`] is. An internal error where the device lacks
/// `sampleRateShading` or `fragmentStoresAndAtomics`.
pub fn center_varyings_check_fragment(device: &Device, width: u32) -> Result<RasterCheck, Error> {
    finish(
        device,
        center_varyings_typed_module(width)?,
        "centre varyings check",
    )
}

/// The coverage counts check's fragment stage of `device`, for a target `width` wide (the module
/// docs), built as [`raster_check_vertex`] is. An internal error where the device lacks
/// `fragmentStoresAndAtomics`.
pub fn coverage_count_check_fragment(device: &Device, width: u32) -> Result<RasterCheck, Error> {
    finish(
        device,
        coverage_count_typed_module(width)?,
        "coverage counts check",
    )
}

/// The early tests check's fragment stage of `device`, for a target `width` wide (the module
/// docs), built as [`raster_check_vertex`] is. An internal error where the device lacks
/// `fragmentStoresAndAtomics`.
pub fn early_tests_check_fragment(device: &Device, width: u32) -> Result<RasterCheck, Error> {
    finish(
        device,
        early_tests_typed_module(width)?,
        "early tests check",
    )
}

/// The barycentric pair check's fragment stage of `device` (the module docs), or its control,
/// which reads `BaryCoordKHR` alone, where not `both`; built as [`raster_check_vertex`] is. An
/// internal error where the device lacks `fragmentShaderBarycentric`.
pub fn barycentric_pair_check_fragment(device: &Device, both: bool) -> Result<RasterCheck, Error> {
    finish(
        device,
        barycentric_pair_typed_module(both)?,
        "barycentric pair check",
    )
}

/// The cube gradients check's fragment stage of `device` (the module docs), built as
/// [`raster_check_vertex`] is. An internal error where the device lacks
/// `fragmentStoresAndAtomics` or `imageCubeArray`.
pub fn cube_gradients_check_fragment(device: &Device) -> Result<RasterCheck, Error> {
    finish(
        device,
        cube_gradients_typed_module(false)?,
        "cube gradients check",
    )
}

/// The cube gradients check's kernel of `device` (the module docs), built as
/// [`raster_check_vertex`] is. An internal error where the device lacks `imageCubeArray`.
pub fn cube_gradients_check_kernel(device: &Device) -> Result<RasterCheck, Error> {
    finish(
        device,
        cube_gradients_typed_module(true)?,
        "cube gradients kernel",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::VkFeature;
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// Every module is valid on every committed profile that has what it needs; each is
    /// printed where a profile lacks it.
    #[test]
    fn the_raster_check_modules_are_valid_on_every_committed_profile() {
        type Build = fn() -> BResult<TypedModule>;
        let modules: [(&str, Build, &[VkFeature]); 10] = [
            ("vertex", vertex_typed_module, &[]),
            (
                "fragment depth",
                || fragcoord_depth_typed_module(24),
                &[
                    VkFeature::SampleRateShading,
                    VkFeature::FragmentStoresAndAtomics,
                ],
            ),
            ("stored depths", || depth_samples_typed_module(24), &[]),
            (
                "centre varyings",
                || center_varyings_typed_module(8),
                &[
                    VkFeature::SampleRateShading,
                    VkFeature::FragmentStoresAndAtomics,
                ],
            ),
            (
                "coverage counts",
                || coverage_count_typed_module(48),
                &[VkFeature::FragmentStoresAndAtomics],
            ),
            (
                "early tests",
                || early_tests_typed_module(16),
                &[VkFeature::FragmentStoresAndAtomics],
            ),
            (
                "barycentric pair",
                || barycentric_pair_typed_module(true),
                &[VkFeature::FragmentShaderBarycentric],
            ),
            (
                "barycentric control",
                || barycentric_pair_typed_module(false),
                &[VkFeature::FragmentShaderBarycentric],
            ),
            (
                "cube gradients",
                || cube_gradients_typed_module(false),
                &[
                    VkFeature::FragmentStoresAndAtomics,
                    VkFeature::ImageCubeArray,
                ],
            ),
            (
                "cube gradients kernel",
                || cube_gradients_typed_module(true),
                &[VkFeature::ImageCubeArray],
            ),
        ];
        for (stem, p) in committed_profiles().expect("profiles") {
            for (name, build, needs) in &modules {
                let features = &p.enabled().features;
                match needs.iter().find(|f| !features.contains(f)) {
                    None => {
                        assert_valid(build().expect("module"), &p);
                    }
                    Some(missing) => eprintln!("{stem}: no {missing:?}, the {name} check skipped"),
                }
            }
        }
    }
}
