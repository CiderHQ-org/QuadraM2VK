//! Generated companion stages of authored graphics cases.
//!
//! A fragment case has no vertex function of its own, and a vertex case observed through its
//! position or a varying has no fragment function that observes it. Both executors, the Metal
//! oracle and the engine candidate, draw such a case with the same generated stage, a
//! [`Companion`]. The checker derives it once from the entry's decoded AIR ([`derive()`]): the
//! ABI's varying user keys, which are Metal's interface-matching identity, and the IR types of
//! the values that cross the interface. [`Companion::air_text`] writes it as AIR, and each executor
//! compiles that module through its own path: the oracle assembles the typed-pointer spelling
//! with `metal-as` and loads it like any other library, the engine candidate assembles the
//! opaque-pointer spelling with `llvm-as` and compiles it in its compile worker beside the case's
//! own library. The spellings differ only in how pointers are spelled: in metadata naming a
//! function, and in a mesh function's output handle and payload, so both executors run one
//! program.
//!
//! - **Passthrough vertex** (a fragment case): Metal's fullscreen triangle, `a = (vid & 1, vid >>
//!   1)`, position `(4a - 1, 0, 1)`, covering clip space with the vertices `(-1, -1)`, `(3, -1)`,
//!   `(-1, 3)`; every varying the fragment function declares, under its user key and IR type,
//!   with the value [`PassthroughValue`] names (from `uv = 2a`); and the viewport and render
//!   target array indices as 0 when the fragment reads them.
//! - **Observer fragment** (a vertex case with a vertex observation): reads the position, or the
//!   varying the observation's location names (the vertex function's varyings in output order),
//!   and writes it to render target 0 padded to four lanes as `(v, 0, 0, 1)` fills them.
//! - **Mesh observer fragment** (a rasterizing mesh case that states a mesh observation): the same
//!   fragment function over the mesh function's outputs: the position, or the vertex or primitive
//!   data field the observation names, under its user key (Metal links a mesh function's vertex
//!   and primitive data to fragment inputs by user key alike), the type the mesh function writes
//!   it with (`air.set_vertex_data_mesh.<T>`, `air.set_primitive_data_mesh.<T>`), and a primitive
//!   data field read flat, its one value over each primitive.
//! - **Launch observer** (an object case that states [`MeshObservation::Launch`]): a mesh
//!   function in the mesh partner's place, and the observer fragment of the primitive data it
//!   writes. Each mesh threadgroup the object function launches draws, from its thread 0, one row
//!   of render target 0 (`x + X * (y + Y * z)` for its position `(x, y, z)` in the launched grid
//!   of `(X, Y, Z)` threadgroups): pixel 0 holds `(X, Y, Z, row)`, and pixel `1 + k` the payload's
//!   words `4k` to `4k + 3`, read at its declared size, a word past it 0. Each pixel is one
//!   triangle over its centre and no other pixel's, as the owned `mesh_draws` fixture draws, whose
//!   one primitive data field (a `uint4`, read flat) the observer writes.
//! - **Payload object** (a mesh case that states an [`ObjectLaunch`]): an object function in the
//!   object partner's place. Thread 0 of each object threadgroup writes the authored payload's
//!   words and sets the authored grid of mesh threadgroups, so every object threadgroup launches
//!   the same mesh threadgroups over the same payload.
//!
//! The values are the ones the oracle's generated MSL wrote through `ORACLE_ABI` v34 (and that
//! the old engine's companions mirror), rule for rule, so a re-recorded Metal observation of an
//! installed case is expected to repeat its v34 output byte for byte.

mod air;
#[cfg(test)]
pub(crate) mod tests;

use crate::air::reflect::TextureComponent;
use crate::case::{
    AuthoredCase, MeshObservation, ObjectLaunch, Stage, TextureFormat, VertexObservation,
};
use crate::check::interface::AirInterface;
use crate::executor_contract::mesh::{launch_payload_words, object_payload_bytes};
use metal2vulkan_air::abi::{
    AirAbi, Builtin, EntryPoint, MeshField, MeshFieldRole, MeshOutputs, OutputRole, ParamRole,
    UserKey,
};
use metal2vulkan_air::intrinsics::{AirIntrinsic, Elem, Mesh, NumTy};
use metal2vulkan_air::ir::{Callee, Const, InstKind, Module, Type, TypeId, Value};
use std::collections::BTreeSet;

pub use air::Spelling;

/// The entry of a generated passthrough vertex function.
pub const PASSTHROUGH_VERTEX_ENTRY: &str = "metal2vulkan_passthrough_vertex";
/// The entry of a generated observer fragment function.
pub const OBSERVER_FRAGMENT_ENTRY: &str = "metal2vulkan_observer_fragment";
/// The entry of a generated launch observer's mesh function.
pub const OBSERVER_MESH_ENTRY: &str = "metal2vulkan_observer_mesh";
/// The entry of a generated payload object function.
pub const PAYLOAD_OBJECT_ENTRY: &str = "metal2vulkan_payload_object";
/// The user key of the launch observer's primitive data field, which its mesh function writes and
/// its observer fragment reads: the key Metal's compiler gives a `uint4` field named `observed`.
const LAUNCH_KEY: &[u8] = b"generated(8observedDv4_j)";

/// The generated stages an authored graphics case draws with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Companion {
    /// The vertex function of a fragment case.
    PassthroughVertex(PassthroughVertex),
    /// The fragment function of a vertex case observed through its position or a varying.
    ObserverFragment(ObserverFragment),
    /// The stages a mesh render pipeline's draw takes from its observation.
    MeshStages(MeshStages),
}

impl Companion {
    /// The entry of the companion's first function in pipeline order: its one function, but for
    /// mesh stages, whose others [`stage_entry`](Self::stage_entry) names.
    pub fn entry(&self) -> &'static str {
        match self {
            Companion::PassthroughVertex(_) => PASSTHROUGH_VERTEX_ENTRY,
            Companion::ObserverFragment(_) => OBSERVER_FRAGMENT_ENTRY,
            Companion::MeshStages(stages) => match (&stages.object, &stages.mesh) {
                (Some(_), _) => PAYLOAD_OBJECT_ENTRY,
                (None, Some(_)) => OBSERVER_MESH_ENTRY,
                (None, None) => OBSERVER_FRAGMENT_ENTRY,
            },
        }
    }

    /// The entry of the companion's function at `stage`, `None` where it generates none.
    pub fn stage_entry(&self, stage: Stage) -> Option<&'static str> {
        match (self, stage) {
            (Companion::PassthroughVertex(_), Stage::Vertex) => Some(PASSTHROUGH_VERTEX_ENTRY),
            (Companion::ObserverFragment(_), Stage::Fragment) => Some(OBSERVER_FRAGMENT_ENTRY),
            (Companion::MeshStages(stages), Stage::Object) => {
                stages.object.as_ref().map(|_| PAYLOAD_OBJECT_ENTRY)
            }
            (Companion::MeshStages(stages), Stage::Mesh) => {
                stages.mesh.as_ref().map(|_| OBSERVER_MESH_ENTRY)
            }
            (Companion::MeshStages(stages), Stage::Fragment) => {
                stages.fragment.as_ref().map(|_| OBSERVER_FRAGMENT_ENTRY)
            }
            _ => None,
        }
    }

    /// The companion as an AIR module in `spelling`.
    pub fn air_text(&self, spelling: Spelling) -> String {
        air::module_text(self, spelling)
    }

    /// The engine candidate's library: [`Spelling::Opaque`] assembled by `llvm-as`.
    pub fn opaque_bitcode(&self) -> Result<Vec<u8>, String> {
        crate::check::loop_free::assemble(&self.air_text(Spelling::Opaque))
            .map_err(|error| format!("assemble the generated companion: {error}"))
    }
}

/// A fullscreen-triangle vertex function that writes every varying a fragment function reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassthroughVertex {
    /// One per fragment input, in the fragment function's parameter order.
    pub varyings: Vec<PassthroughVarying>,
    /// Write `[[viewport_array_index]]` 0: the fragment function reads it.
    pub viewport_array_index: bool,
    /// Write `[[render_target_array_index]]` 0: the fragment function reads it.
    pub render_target_array_index: bool,
}

/// One varying the passthrough vertex writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassthroughVarying {
    /// The fragment input's user key, which Metal links the stages by.
    pub key: UserKey,
    pub ty: VaryingType,
    /// The fragment input's declared type name, repeated as the vertex output's
    /// `air.arg_type_name` (Metal names the output's type there; it decides nothing here).
    pub type_name: String,
    pub value: PassthroughValue,
}

/// What a passthrough varying holds at each vertex of the fullscreen triangle, with `uv = 2a`
/// (`(0, 0)`, `(2, 0)`, `(0, 2)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassthroughValue {
    /// A float or half scalar: 0.25.
    Quarter,
    /// Two float or half lanes: `uv`.
    Uv,
    /// Three float or half lanes: `(uv, 0.5)`, or `(uv, 1)` when `z_one`.
    UvZ { z_one: bool },
    /// Four float or half lanes: `(uv, 0.5, 1)`.
    UvHalfOne,
    /// Integer lanes 1, 2, 3, 4.
    Counting,
    /// Boolean lanes true, false, true, false.
    Alternating,
    /// The case's own value (`AuthoredCase::varying_values`) at every vertex: each lane's bits at
    /// the scalar's width, of which the varying's lane count are used.
    Authored { lanes: [u32; 4] },
}

/// The generated stages of an object or mesh case's mesh render pipeline, each in the place of a
/// partner its draw does not name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshStages {
    /// The object function of a mesh case's object launch.
    pub object: Option<PayloadObject>,
    /// The mesh function of an object case's launch observation.
    pub mesh: Option<LaunchObserver>,
    /// The observer fragment: of a mesh case's observation, or of the launch observer's rows.
    pub fragment: Option<ObserverFragment>,
}

/// A mesh function that draws what an object function launches into render target 0, a row for
/// each mesh threadgroup ([`MeshObservation::Launch`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchObserver {
    /// The object function's payload, which the mesh function declares as its own; `None` where
    /// the object function declares none.
    pub payload: Option<LaunchPayload>,
    /// Render target 0's width and height, which place each row's pixels.
    pub target: [u32; 2],
}

/// The payload a launch observer reads: the object function's declared size in 4-byte words, and
/// its declared alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchPayload {
    pub words: u32,
    pub align: u32,
}

/// An object function that writes a payload and launches a grid of mesh threadgroups
/// ([`ObjectLaunch`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PayloadObject {
    /// The payload's words, which thread 0 of each object threadgroup writes.
    pub words: Vec<u32>,
    /// The payload's declared alignment: the mesh function's, at least 4.
    pub align: u32,
    /// The mesh threadgroups each object threadgroup launches.
    pub mesh_threadgroups: [u32; 3],
}

impl LaunchObserver {
    /// The pixels of each row: the grid's, and one for each four payload words.
    pub fn pixels(&self) -> u32 {
        1 + self.payload.map_or(0, |payload| payload.words.div_ceil(4))
    }
}

/// A fragment function that writes one vertex value to render target 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObserverFragment {
    pub input: ObservedValue,
    pub output: ObserverOutput,
}

/// The vertex value an observer reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObservedValue {
    /// `[[position]]`, as Metal's rasterizer delivers it (center, no perspective).
    Position,
    /// The varying of user key `key`, interpolated flat for an integer or boolean or a value
    /// `per_primitive`, else at the center with perspective (MSL's default).
    Varying {
        key: UserKey,
        ty: VaryingType,
        /// The vertex output's declared type name, repeated as the fragment input's.
        type_name: String,
        /// A mesh function's primitive data: one value over each primitive.
        per_primitive: bool,
    },
}

/// The texel class render target 0 receives, and so how an integer lane extends to 32 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObserverOutput {
    /// `float4`: float lanes as they are, half lanes widened.
    Float,
    /// `int4`: integer lanes sign-extended.
    Sint,
    /// `uint4`: integer lanes zero-extended, booleans as 0 or 1.
    Uint,
}

/// A varying's IR type: one of the scalars a companion writes, with 1 to 4 lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaryingType {
    pub scalar: Scalar,
    pub lanes: u32,
}

/// The scalar of a [`VaryingType`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    F32,
    F16,
    I32,
    I16,
    Bool,
}

impl Scalar {
    fn is_float(self) -> bool {
        matches!(self, Scalar::F32 | Scalar::F16)
    }
}

impl VaryingType {
    /// The varying type of IR type `ty`, `None` for any other.
    fn of(module: &Module, ty: TypeId) -> Option<VaryingType> {
        let types = module.types();
        let scalar = |ty: TypeId| match types.get(ty)? {
            Type::Float => Some(Scalar::F32),
            Type::Half => Some(Scalar::F16),
            Type::Int(32) => Some(Scalar::I32),
            Type::Int(16) => Some(Scalar::I16),
            Type::Int(1) => Some(Scalar::Bool),
            _ => None,
        };
        let (scalar, lanes) = match types.get(ty)? {
            Type::Vector { elem, len } => (scalar(*elem)?, *len),
            _ => (scalar(ty)?, 1),
        };
        (1..=4)
            .contains(&lanes)
            .then_some(VaryingType { scalar, lanes })
    }

    /// The MSL spelling of this type with a signless integer read as signed: the type name the
    /// metadata states when the AIR names none.
    fn spelled(self) -> String {
        let base = match self.scalar {
            Scalar::F32 => "float",
            Scalar::F16 => "half",
            Scalar::I32 => "int",
            Scalar::I16 => "short",
            Scalar::Bool => "bool",
        };
        if self.lanes == 1 {
            base.into()
        } else {
            format!("{base}{}", self.lanes)
        }
    }
}

/// The companion `case` draws with, derived from its entry in `module` (whose decoded ABI is
/// `abi` and whose interface for the case is `interface`): a [`PassthroughVertex`] for a fragment
/// case, an [`ObserverFragment`] for a vertex case with a vertex observation, [`MeshStages`] for
/// an object or mesh case with a mesh observation or an object launch, `None` for any other case.
/// An `Err` names what no companion can supply (a varying of a type the companions do not write,
/// an observation of something the vertex or mesh function does not output, a payload the launch
/// observer cannot read or a launch that holds less than the mesh function reads), so the case
/// cannot be executed as authored.
pub fn derive(
    case: &AuthoredCase,
    module: &Module,
    abi: &AirAbi,
    interface: &AirInterface,
) -> Result<Option<Companion>, String> {
    let observation = match (case.stage, case.vertex_observation) {
        (Stage::Fragment, _) => None,
        (Stage::Vertex, Some(observation)) => Some(observation),
        (Stage::Mesh | Stage::Object, _) => {
            let Some(draw) = case.mesh_draw.as_ref().filter(|draw| draw.generates()) else {
                return Ok(None);
            };
            let entry = entry_point(module, abi, &case.entry)?;
            let stages = match (case.stage, draw.observation) {
                (Stage::Object, Some(MeshObservation::Launch)) => MeshStages {
                    object: None,
                    mesh: Some(launch_observer(case, entry)?),
                    fragment: Some(ObserverFragment {
                        input: ObservedValue::Varying {
                            key: UserKey(LAUNCH_KEY.into()),
                            ty: VaryingType {
                                scalar: Scalar::I32,
                                lanes: 4,
                            },
                            type_name: "uint4".into(),
                            per_primitive: true,
                        },
                        output: ObserverOutput::Uint,
                    }),
                },
                (Stage::Mesh, observation) => MeshStages {
                    object: draw
                        .object_launch
                        .as_ref()
                        .map(|launch| payload_object(entry, launch))
                        .transpose()?,
                    mesh: None,
                    fragment: observation
                        .map(|observation| mesh_observer(case, module, entry, observation))
                        .transpose()?,
                },
                (_, observation) => {
                    return Err(format!(
                        "an object case's observation {observation:?}: an object case observes \
                         its launch"
                    ))
                }
            };
            return Ok(Some(Companion::MeshStages(stages)));
        }
        (Stage::Vertex, None) | (Stage::Kernel, _) => return Ok(None),
    };
    match observation {
        None => passthrough_vertex(module, abi, &case.entry)
            .and_then(|vertex| with_authored_values(vertex, case))
            .map(Companion::PassthroughVertex),
        Some(observation) => {
            let entry = entry_point(module, abi, &case.entry)?;
            let signature = signature(module, entry)?;
            observer(case, module, entry, &signature, interface, observation)
                .map(Companion::ObserverFragment)
        }
    }
    .map(Some)
}

/// The passthrough vertex companion of fragment function `entry` in `module` (whose decoded ABI
/// is `abi`): what [`derive()`] gives a fragment case, and what an L4 row of a fragment function
/// draws with (`acceptance::row`). An `Err` names a varying no companion writes.
pub fn passthrough_vertex(
    module: &Module,
    abi: &AirAbi,
    entry: &str,
) -> Result<PassthroughVertex, String> {
    let entry = entry_point(module, abi, entry)?;
    let signature = signature(module, entry)?;
    passthrough(module, entry, &signature)
}

/// `vertex` with the case's authored varying values in place of the defaults. An `Err` names a
/// value no fragment input can take: a key the fragment function does not declare, or lanes that do
/// not fit the input's IR type.
fn with_authored_values(
    mut vertex: PassthroughVertex,
    case: &AuthoredCase,
) -> Result<PassthroughVertex, String> {
    for authored in &case.varying_values {
        let varying = vertex
            .varyings
            .iter_mut()
            .find(|varying| varying.key.0.as_ref() == authored.key.as_bytes())
            .ok_or_else(|| {
                format!(
                    "varying value {:?} names no varying the fragment function reads",
                    authored.key
                )
            })?;
        varying.value = authored_value(varying.ty, &authored.key, &authored.lanes)?;
    }
    Ok(vertex)
}

/// The passthrough value of `lanes` for a varying of type `ty` (user key `key`).
fn authored_value(ty: VaryingType, key: &str, lanes: &[u32]) -> Result<PassthroughValue, String> {
    if lanes.len() != ty.lanes as usize {
        return Err(format!(
            "varying value {key:?} has {} lanes for a {}-lane {} varying",
            lanes.len(),
            ty.lanes,
            ty.spelled()
        ));
    }
    let limit = match ty.scalar {
        Scalar::F32 | Scalar::I32 => u32::MAX,
        Scalar::F16 | Scalar::I16 => u32::from(u16::MAX),
        Scalar::Bool => 1,
    };
    if let Some(lane) = lanes.iter().find(|&&lane| lane > limit) {
        return Err(format!(
            "varying value {key:?} has lane bits {lane:#x}, wider than its {} varying's lanes",
            ty.spelled()
        ));
    }
    let mut bits = [0; 4];
    bits[..lanes.len()].copy_from_slice(lanes);
    Ok(PassthroughValue::Authored { lanes: bits })
}

/// The entry point of `abi` whose function is named `name`.
fn entry_point<'a>(module: &Module, abi: &'a AirAbi, name: &str) -> Result<&'a EntryPoint, String> {
    abi.entries
        .iter()
        .find(|ep| {
            module
                .func(ep.func)
                .and_then(|f| module.symbols().get(f.name))
                == Some(name.as_bytes())
        })
        .ok_or_else(|| format!("AIR declares no entry point {name:?}"))
}

/// An entry's parameter types and the type of each of its outputs, in order.
struct Signature {
    params: Vec<TypeId>,
    outputs: Vec<TypeId>,
}

fn signature(module: &Module, entry: &EntryPoint) -> Result<Signature, String> {
    let types = module.types();
    let function = module
        .func(entry.func)
        .ok_or("the entry point names no function")?;
    let Some(Type::Function { ret, params, .. }) = types.get(function.ty) else {
        return Err("the entry point's function has no function type".into());
    };
    let outputs = match types.get(*ret) {
        Some(Type::Struct { fields, .. }) => fields.to_vec(),
        Some(Type::Void) => Vec::new(),
        _ => vec![*ret],
    };
    if outputs.len() != entry.outputs.len() {
        return Err(format!(
            "the entry returns {} values for {} outputs",
            outputs.len(),
            entry.outputs.len()
        ));
    }
    Ok(Signature {
        params: params.to_vec(),
        outputs,
    })
}

/// The declared type name `name`, or the IR type's spelling when the AIR names none.
fn type_name(name: Option<&metal2vulkan::error::DiagName>, ty: VaryingType) -> String {
    name.map_or_else(|| ty.spelled(), ToString::to_string)
}

fn passthrough(
    module: &Module,
    entry: &EntryPoint,
    signature: &Signature,
) -> Result<PassthroughVertex, String> {
    let mut inputs = Vec::new();
    let mut viewport_array_index = false;
    let mut render_target_array_index = false;
    for binding in &entry.params {
        match &binding.role {
            ParamRole::FragmentInput(varying) => {
                let ir = signature
                    .params
                    .get(binding.param as usize)
                    .copied()
                    .ok_or_else(|| format!("fragment input {} has no parameter", binding.param))?;
                let ty = VaryingType::of(module, ir).ok_or_else(|| {
                    format!(
                        "fragment input {} has a type no generated passthrough vertex writes \
                         (float, half, 32- and 16-bit integers, and bool, of 1 to 4 lanes)",
                        binding.param
                    )
                })?;
                inputs.push((
                    varying.key.clone(),
                    ty,
                    type_name(binding.ty.name.as_ref(), ty),
                ));
            }
            ParamRole::Builtin(Builtin::ViewportArrayIndex) => viewport_array_index = true,
            ParamRole::Builtin(Builtin::RenderTargetArrayIndex) => render_target_array_index = true,
            _ => {}
        }
    }
    let float3 = |ty: &VaryingType| ty.scalar == Scalar::F32 && ty.lanes == 3;
    let distinct_float3 = inputs.iter().filter(|(_, ty, _)| float3(ty)).count() >= 2
        && normalizes_a_float3_difference(module);
    let varyings = inputs
        .into_iter()
        .enumerate()
        .map(|(ordinal, (key, ty, type_name))| PassthroughVarying {
            key,
            ty,
            type_name,
            value: passthrough_value(ty, distinct_float3 && ordinal > 0),
        })
        .collect();
    Ok(PassthroughVertex {
        varyings,
        viewport_array_index,
        render_target_array_index,
    })
}

/// The value of a passthrough varying of type `ty`; `z_one` moves a three-lane float or half's
/// third lane from 0.5 to 1.
fn passthrough_value(ty: VaryingType, z_one: bool) -> PassthroughValue {
    match (ty.scalar, ty.lanes) {
        (Scalar::I32 | Scalar::I16, _) => PassthroughValue::Counting,
        (Scalar::Bool, _) => PassthroughValue::Alternating,
        (_, 1) => PassthroughValue::Quarter,
        (_, 2) => PassthroughValue::Uv,
        (_, 3) => PassthroughValue::UvZ { z_one },
        _ => PassthroughValue::UvHalfOne,
    }
}

/// The oracle's rule for three-lane float varyings (kept from its MSL passthrough so recorded
/// observations stay comparable): with two or more of them, every one after the first varying
/// gets `z = 1` when the module declares `air.fast_rsqrt.f32` and subtracts `<3 x float>` values,
/// the compiled shape of `normalize(a - b)` over two varyings, which equal varyings would make
/// `normalize(0)`. Structural: an `air.*` ABI symbol and IR types.
fn normalizes_a_float3_difference(module: &Module) -> bool {
    let types = module.types();
    let float3 = |ty: TypeId| {
        matches!(types.get(ty), Some(Type::Vector { elem, len: 3 })
            if matches!(types.get(*elem), Some(Type::Float)))
    };
    let functions = &module.parts().funcs;
    let declares_fast_rsqrt = functions
        .values()
        .any(|f| module.symbols().get(f.name) == Some(b"air.fast_rsqrt.f32".as_slice()));
    let subtracts_float3 = functions
        .values()
        .filter_map(|f| f.body.as_ref())
        .flat_map(|body| body.insts.values())
        .any(|inst| {
            matches!(
                inst.kind,
                InstKind::BinOp {
                    op: metal2vulkan_air::ir::BinOp::FSub,
                    ..
                }
            ) && float3(inst.ty)
        });
    declares_fast_rsqrt && subtracts_float3
}

/// The format of `case`'s render target 0, which the observer of `what` (a vertex or mesh
/// observation) writes.
fn observed_format(case: &AuthoredCase, what: &str) -> Result<TextureFormat, String> {
    case.render_targets
        .iter()
        .find(|target| target.index == 0)
        .map(|target| target.format)
        .ok_or_else(|| format!("a {what} needs render target 0"))
}

fn observer(
    case: &AuthoredCase,
    module: &Module,
    entry: &EntryPoint,
    signature: &Signature,
    interface: &AirInterface,
    observation: VertexObservation,
) -> Result<ObserverFragment, String> {
    let format = observed_format(case, VERTEX)?;
    let live = |index: usize| {
        interface
            .outputs
            .get(index)
            .is_some_and(|output| output.live)
    };
    match observation {
        VertexObservation::Position => {
            let written = entry.outputs.iter().enumerate().any(|(index, output)| {
                matches!(output.role, OutputRole::Position { .. }) && live(index)
            });
            if !written {
                return Err(
                    "vertex observation selects position, but AIR does not write it".into(),
                );
            }
            expect_format(VERTEX, format, TextureFormat::Rgba32Float)?;
            Ok(ObserverFragment {
                input: ObservedValue::Position,
                output: ObserverOutput::Float,
            })
        }
        VertexObservation::Varying { location } => {
            let (index, output) = entry
                .outputs
                .iter()
                .enumerate()
                .filter(|(_, output)| matches!(output.role, OutputRole::VertexOutput(_)))
                .nth(location as usize)
                .ok_or_else(|| {
                    format!("vertex observation selects undeclared varying location {location}")
                })?;
            if !live(index) {
                return Err(format!(
                    "vertex observation selects varying location {location}, which the case's \
                     function constants remove"
                ));
            }
            let OutputRole::VertexOutput(key) = &output.role else {
                unreachable!("filtered to vertex outputs");
            };
            let ty = signature
                .outputs
                .get(index)
                .and_then(|&ty| VaryingType::of(module, ty))
                .ok_or_else(|| {
                    format!(
                        "vertex varying {location} has a type no generated observer reads \
                         (float, half, 32- and 16-bit integers, and bool, of 1 to 4 lanes)"
                    )
                })?;
            let output_class = observer_output(VERTEX, ty, format)?;
            Ok(ObserverFragment {
                input: ObservedValue::Varying {
                    key: key.clone(),
                    ty,
                    type_name: type_name(output.ty.name.as_ref(), ty),
                    per_primitive: false,
                },
                output: output_class,
            })
        }
    }
}

/// What a mesh function writes of its output (its module's `air.set_*_mesh` calls): its position,
/// and each vertex and primitive data slot with the type each call writes it with.
#[derive(Default)]
struct MeshWrites {
    position: bool,
    vertex_data: BTreeSet<(u32, NumTy)>,
    primitive_data: BTreeSet<(u32, NumTy)>,
}

impl MeshWrites {
    fn of(module: &Module) -> MeshWrites {
        let slot = |value: Option<&Value>| {
            let Some(Value::Const(id)) = value else {
                return None;
            };
            match module.consts().get(*id)?.value {
                Const::Int(slot) => u32::try_from(slot).ok(),
                _ => None,
            }
        };
        let mut writes = MeshWrites::default();
        let calls = module
            .parts()
            .funcs
            .values()
            .filter_map(|function| function.body.as_ref())
            .flat_map(|body| body.insts.values())
            .filter_map(|inst| match &inst.kind {
                InstKind::Call(call) => Some(call),
                _ => None,
            });
        for call in calls {
            let Callee::Air {
                op: AirIntrinsic::Mesh(op),
                ..
            } = &call.callee
            else {
                continue;
            };
            match op {
                Mesh::Position => writes.position = true,
                Mesh::VertexData(ty) | Mesh::PrimitiveData(ty) => {
                    // An `air.set_*_data_mesh` call's slot is its second operand.
                    let Some(slot) = slot(call.args.get(1)) else {
                        continue;
                    };
                    let data = match op {
                        Mesh::VertexData(_) => &mut writes.vertex_data,
                        _ => &mut writes.primitive_data,
                    };
                    data.insert((slot, *ty));
                }
                _ => {}
            }
        }
        writes
    }

    /// The one type the function writes data slot `slot` with (of its primitive data where
    /// `per_primitive`), or why there is none.
    fn data_type(&self, slot: u32, per_primitive: bool) -> Result<NumTy, String> {
        let data = if per_primitive {
            &self.primitive_data
        } else {
            &self.vertex_data
        };
        let types = data
            .iter()
            .filter(|(written, _)| *written == slot)
            .map(|(_, ty)| *ty)
            .collect::<Vec<_>>();
        match types[..] {
            [ty] => Ok(ty),
            [] => Err(format!("the mesh function never writes data slot {slot}")),
            _ => Err(format!(
                "the mesh function writes data slot {slot} with {} types",
                types.len()
            )),
        }
    }
}

/// The varying type of a mesh data field written as `ty`, `None` for a type no observer reads.
fn mesh_data_type(ty: NumTy) -> Option<VaryingType> {
    let scalar = match ty.elem {
        Elem::F32 => Scalar::F32,
        Elem::F16 => Scalar::F16,
        Elem::I32 => Scalar::I32,
        Elem::I16 => Scalar::I16,
        _ => return None,
    };
    (1..=4).contains(&ty.lanes).then_some(VaryingType {
        scalar,
        lanes: ty.lanes,
    })
}

/// The observer fragment of mesh function `entry` (of `module`) for `observation`.
fn mesh_observer(
    case: &AuthoredCase,
    module: &Module,
    entry: &EntryPoint,
    observation: MeshObservation,
) -> Result<ObserverFragment, String> {
    if observation == MeshObservation::Launch {
        return Err(
            "a mesh case's launch observation: it observes an object case's own object function"
                .into(),
        );
    }
    let format = observed_format(case, MESH)?;
    let outputs: &MeshOutputs = entry
        .params
        .iter()
        .find_map(|binding| match &binding.role {
            ParamRole::Mesh(outputs) => Some(outputs.as_ref()),
            _ => None,
        })
        .ok_or("the mesh function declares no mesh output")?;
    let writes = MeshWrites::of(module);
    let gated = |field: &MeshField, what: &str| match field.predicate {
        Some(_) => Err(format!(
            "mesh observation selects {what}, which a function constant gates: not modelled"
        )),
        None => Ok(()),
    };
    match observation {
        MeshObservation::Position => {
            let field = outputs
                .vertex
                .iter()
                .find(|field| matches!(field.role, MeshFieldRole::Position { .. }))
                .filter(|_| writes.position)
                .ok_or(
                    "mesh observation selects position, but the mesh function does not write it",
                )?;
            gated(field, "position")?;
            expect_format(MESH, format, TextureFormat::Rgba32Float)?;
            Ok(ObserverFragment {
                input: ObservedValue::Position,
                output: ObserverOutput::Float,
            })
        }
        MeshObservation::Launch => unreachable!("refused above"),
        MeshObservation::VertexData { index } | MeshObservation::PrimitiveData { index } => {
            let per_primitive = matches!(observation, MeshObservation::PrimitiveData { .. });
            let (fields, what) = if per_primitive {
                (&outputs.primitive, "primitive data")
            } else {
                (&outputs.vertex, "vertex data")
            };
            let (field, slot, key) = fields
                .iter()
                .filter_map(|field| match &field.role {
                    MeshFieldRole::VertexData { slot, key }
                    | MeshFieldRole::PrimitiveData { slot, key } => Some((field, *slot, key)),
                    _ => None,
                })
                .nth(index as usize)
                .ok_or_else(|| {
                    format!("mesh observation selects undeclared {what} field {index}")
                })?;
            gated(field, &format!("{what} field {index}"))?;
            let written = writes.data_type(slot, per_primitive)?;
            let ty = mesh_data_type(written).ok_or_else(|| {
                format!(
                    "mesh {what} field {index} has a type no generated observer reads (float, \
                     half, 32- and 16-bit integers, of 1 to 4 lanes)"
                )
            })?;
            Ok(ObserverFragment {
                input: ObservedValue::Varying {
                    key: key.clone(),
                    ty,
                    type_name: type_name(field.ty.name.as_ref(), ty),
                    per_primitive,
                },
                output: observer_output(MESH, ty, format)?,
            })
        }
    }
}

/// The launch observer of object function `entry` for `case`: the payload it declares, read at
/// its declared size in whole words ([`launch_payload_words`]), into an `RGBA32Uint` render target
/// 0 wide enough for a row's pixels.
fn launch_observer(case: &AuthoredCase, entry: &EntryPoint) -> Result<LaunchObserver, String> {
    let format = observed_format(case, LAUNCH)?;
    expect_format(LAUNCH, format, TextureFormat::Rgba32Uint)?;
    let payload = entry
        .params
        .iter()
        .find(|binding| matches!(binding.role, ParamRole::Payload { .. }));
    if payload.is_some_and(|binding| binding.predicate.is_some()) {
        return Err(
            "launch observation of a payload a function constant gates: not modelled".into(),
        );
    }
    let words = launch_payload_words(payload.map(|binding| binding.ty.size))?;
    let target = case
        .render_targets
        .iter()
        .find(|target| target.index == 0)
        .map(|target| target.dimensions)
        .ok_or("a launch observation needs render target 0")?;
    let observer = LaunchObserver {
        payload: payload.map(|binding| LaunchPayload {
            words,
            align: binding.ty.align.unwrap_or(4),
        }),
        target,
    };
    if target[0] < observer.pixels() {
        return Err(format!(
            "launch observation draws {} pixels a row, and render target 0 is {} wide",
            observer.pixels(),
            target[0]
        ));
    }
    Ok(observer)
}

/// The payload object of mesh function `entry` for `launch`: its payload's words, at least as many
/// bytes as the mesh function reads ([`object_payload_bytes`]), at the mesh function's declared
/// alignment.
fn payload_object(entry: &EntryPoint, launch: &ObjectLaunch) -> Result<PayloadObject, String> {
    let bytes = launch
        .payload()
        .ok_or("mesh_draw.object_launch.payload_b64 does not decode")?;
    let read = entry
        .params
        .iter()
        .find(|binding| matches!(binding.role, ParamRole::Payload { .. }));
    if read.is_some_and(|binding| binding.predicate.is_some()) {
        return Err("object launch for a payload a function constant gates: not modelled".into());
    }
    if let Some(binding) = read {
        let size = object_payload_bytes(binding.ty.size)?;
        if (bytes.len() as u64) < u64::from(size) {
            return Err(format!(
                "the object launch's payload holds {} bytes, and the mesh function reads {size}",
                bytes.len()
            ));
        }
    }
    Ok(PayloadObject {
        words: bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect(),
        align: read
            .and_then(|binding| binding.ty.align)
            .unwrap_or(4)
            .max(4),
        mesh_threadgroups: launch.mesh_threadgroups,
    })
}

/// The output of an observer of a `ty` varying into a `format` render target: floats into
/// `RGBA32Float`, booleans into `RGBA32Uint`, and integers into `RGBA32Sint` or `RGBA32Uint`,
/// whose class states the sign the IR does not carry.
fn observer_output(
    what: &str,
    ty: VaryingType,
    format: TextureFormat,
) -> Result<ObserverOutput, String> {
    let output = match (ty.scalar, format.component_class()) {
        (scalar, _) if scalar.is_float() => {
            expect_format(what, format, TextureFormat::Rgba32Float)?;
            ObserverOutput::Float
        }
        (Scalar::Bool, _) => {
            expect_format(what, format, TextureFormat::Rgba32Uint)?;
            ObserverOutput::Uint
        }
        (_, TextureComponent::Sint) => {
            expect_format(what, format, TextureFormat::Rgba32Sint)?;
            ObserverOutput::Sint
        }
        _ => {
            expect_format(what, format, TextureFormat::Rgba32Uint)?;
            ObserverOutput::Uint
        }
    };
    Ok(output)
}

/// The observations an observer fragment reads, as its errors name them.
const VERTEX: &str = "vertex observation";
const MESH: &str = "mesh observation";
const LAUNCH: &str = "launch observation";

fn expect_format(what: &str, format: TextureFormat, expected: TextureFormat) -> Result<(), String> {
    if format == expected {
        Ok(())
    } else {
        Err(format!(
            "{what} requires render target format {expected:?}, got {format:?}"
        ))
    }
}
