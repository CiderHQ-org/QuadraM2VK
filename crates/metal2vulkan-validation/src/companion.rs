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
//! own library. The spellings differ only in how metadata names the function (no body uses a
//! pointer), so both executors run one program.
//!
//! - **Passthrough vertex** (a fragment case): Metal's fullscreen triangle, `a = (vid & 1, vid >>
//!   1)`, position `(4a - 1, 0, 1)`, covering clip space with the vertices `(-1, -1)`, `(3, -1)`,
//!   `(-1, 3)`; every varying the fragment function declares, under its user key and IR type,
//!   with the value [`PassthroughValue`] names (from `uv = 2a`); and the viewport and render
//!   target array indices as 0 when the fragment reads them.
//! - **Observer fragment** (a vertex case with a vertex observation): reads the position, or the
//!   varying the observation's location names (the vertex function's varyings in output order),
//!   and writes it to render target 0 padded to four lanes as `(v, 0, 0, 1)` fills them.
//!
//! The values are the ones the oracle's generated MSL wrote through `ORACLE_ABI` v34 (and that
//! the old engine's companions mirror), rule for rule, so a re-recorded Metal observation of an
//! installed case is expected to repeat its v34 output byte for byte.

mod air;
#[cfg(test)]
pub(crate) mod tests;

use crate::air::reflect::TextureComponent;
use crate::case::{AuthoredCase, Stage, TextureFormat, VertexObservation};
use crate::check::interface::AirInterface;
use metal2vulkan_air::abi::{AirAbi, Builtin, EntryPoint, OutputRole, ParamRole, UserKey};
use metal2vulkan_air::ir::{InstKind, Module, Type, TypeId};

pub use air::Spelling;

/// The entry of a generated passthrough vertex function.
pub const PASSTHROUGH_VERTEX_ENTRY: &str = "metal2vulkan_passthrough_vertex";
/// The entry of a generated observer fragment function.
pub const OBSERVER_FRAGMENT_ENTRY: &str = "metal2vulkan_observer_fragment";

/// The generated stage an authored graphics case draws with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Companion {
    /// The vertex function of a fragment case.
    PassthroughVertex(PassthroughVertex),
    /// The fragment function of a vertex case observed through its position or a varying.
    ObserverFragment(ObserverFragment),
}

impl Companion {
    /// The companion's entry point.
    pub fn entry(&self) -> &'static str {
        match self {
            Companion::PassthroughVertex(_) => PASSTHROUGH_VERTEX_ENTRY,
            Companion::ObserverFragment(_) => OBSERVER_FRAGMENT_ENTRY,
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
    /// The varying of user key `key`, interpolated flat for an integer or boolean, else at the
    /// center with perspective (MSL's default).
    Varying {
        key: UserKey,
        ty: VaryingType,
        /// The vertex output's declared type name, repeated as the fragment input's.
        type_name: String,
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
/// case, an [`ObserverFragment`] for a vertex case with a vertex observation, `None` for any
/// other case. An `Err` names what no companion can supply (a varying of a type the companions do
/// not write, an observation of something the vertex function does not output), so the case
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
        (Stage::Vertex, None) | (Stage::Kernel | Stage::Object | Stage::Mesh, _) => {
            return Ok(None)
        }
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

fn observer(
    case: &AuthoredCase,
    module: &Module,
    entry: &EntryPoint,
    signature: &Signature,
    interface: &AirInterface,
    observation: VertexObservation,
) -> Result<ObserverFragment, String> {
    let format = case
        .render_targets
        .iter()
        .find(|target| target.index == 0)
        .map(|target| target.format)
        .ok_or("a vertex observation needs render target 0")?;
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
            expect_format(format, TextureFormat::Rgba32Float)?;
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
            let output_class = observer_output(ty, format)?;
            Ok(ObserverFragment {
                input: ObservedValue::Varying {
                    key: key.clone(),
                    ty,
                    type_name: type_name(output.ty.name.as_ref(), ty),
                },
                output: output_class,
            })
        }
    }
}

/// The output of an observer of a `ty` varying into a `format` render target: floats into
/// `RGBA32Float`, booleans into `RGBA32Uint`, and integers into `RGBA32Sint` or `RGBA32Uint`,
/// whose class states the sign the IR does not carry.
fn observer_output(ty: VaryingType, format: TextureFormat) -> Result<ObserverOutput, String> {
    let output = match (ty.scalar, format.component_class()) {
        (scalar, _) if scalar.is_float() => {
            expect_format(format, TextureFormat::Rgba32Float)?;
            ObserverOutput::Float
        }
        (Scalar::Bool, _) => {
            expect_format(format, TextureFormat::Rgba32Uint)?;
            ObserverOutput::Uint
        }
        (_, TextureComponent::Sint) => {
            expect_format(format, TextureFormat::Rgba32Sint)?;
            ObserverOutput::Sint
        }
        _ => {
            expect_format(format, TextureFormat::Rgba32Uint)?;
            ObserverOutput::Uint
        }
    };
    Ok(output)
}

fn expect_format(format: TextureFormat, expected: TextureFormat) -> Result<(), String> {
    if format == expected {
        Ok(())
    } else {
        Err(format!(
            "vertex observation requires render target format {expected:?}, got {format:?}"
        ))
    }
}
