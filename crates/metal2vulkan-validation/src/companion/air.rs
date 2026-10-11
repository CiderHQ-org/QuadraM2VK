//! The AIR text of a [`Companion`]: one entry point, no pointers in any body, and the module
//! metadata Metal's `metal-opt` verification and the product's ABI decoder both require.
//!
//! The module mirrors what the Metal compiler emits for the oracle's former MSL companions: the
//! same data layout and triple, `air.convert` for the vertex id, `fptrunc`/`fpext` for half
//! lanes, a packed struct of outputs, and the compile options the Metal compiler sets by default.

use super::{
    Companion, ObservedValue, ObserverFragment, ObserverOutput, PassthroughValue,
    PassthroughVertex, Scalar, VaryingType,
};
use metal2vulkan_air::abi::UserKey;
use std::fmt::Write as _;

/// How the module is spelled for the toolchain that assembles it. Only the metadata's reference
/// to the entry function differs: no body uses a pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spelling {
    /// Typed pointers, for Apple's `metal-as` (the Metal oracle).
    Typed,
    /// Opaque pointers, for `llvm-as` of LLVM 17 and later (the engine candidate).
    Opaque,
}

const DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-\
f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-\
v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";
const TRIPLE: &str = "air64_v28-apple-macosx26.0.0";
/// The module's source name. Without one, an assembler names the module after the scratch file it
/// reads, so the bitcode (and every key hashed from it) would differ between two assemblies of the
/// same companion.
const SOURCE_FILENAME: &str = "metal2vulkan-companion";

/// The module text of `companion` in `spelling`.
pub(super) fn module_text(companion: &Companion, spelling: Spelling) -> String {
    let (what, function, list) = match companion {
        Companion::PassthroughVertex(vertex) => (
            "a passthrough vertex function",
            passthrough(vertex),
            "air.vertex",
        ),
        Companion::ObserverFragment(fragment) => (
            "an observer fragment function",
            observer(fragment),
            "air.fragment",
        ),
    };
    let mut md = Metadata::default();
    let flags = [
        ("air.max_device_buffers", 31),
        ("air.max_constant_buffers", 31),
        ("air.max_threadgroup_buffers", 31),
        ("air.max_textures", 128),
        ("air.max_read_write_textures", 8),
        ("air.max_samplers", 16),
    ]
    .map(|(name, value)| md.node(format!("!{{i32 7, !\"{name}\", i32 {value}}}")));
    let outputs = function
        .outputs
        .iter()
        .map(|node| md.node(node.clone()))
        .collect::<Vec<_>>();
    let inputs = function
        .inputs
        .iter()
        .map(|node| md.node(node.clone()))
        .collect::<Vec<_>>();
    let outputs = md.node(list_of(&outputs));
    let inputs = md.node(list_of(&inputs));
    let reference = match spelling {
        Spelling::Typed => format!(
            "{} ({})* @{}",
            function.ret,
            function.params.join(", "),
            companion.entry()
        ),
        Spelling::Opaque => format!("ptr @{}", companion.entry()),
    };
    let entry = md.node(format!("!{{{reference}, !{outputs}, !{inputs}}}"));
    let options = [
        "air.compile.denorms_disable",
        "air.compile.fast_math_enable",
        "air.compile.framebuffer_fetch_enable",
    ]
    .map(|option| md.node(format!("!{{!\"{option}\"}}")));
    let version = md.node("!{i32 2, i32 8, i32 0}".into());
    let language = md.node("!{!\"Metal\", i32 4, i32 0, i32 0}".into());

    let mut text = format!(
        "; metal2vulkan generated companion (WP-V3c): {what}.\n\
         source_filename = \"{SOURCE_FILENAME}\"\n\
         target datalayout = \"{DATA_LAYOUT}\"\n\
         target triple = \"{TRIPLE}\"\n\n"
    );
    let _ = writeln!(
        text,
        "define {} @{}({}) {{",
        function.ret,
        companion.entry(),
        function
            .params
            .iter()
            .zip(&function.param_names)
            .map(|(ty, name)| format!("{ty} %{name}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for line in &function.body {
        let _ = writeln!(text, "  {line}");
    }
    text.push_str("}\n");
    for declaration in &function.declarations {
        let _ = writeln!(text, "\n{declaration}");
    }
    text.push('\n');
    let _ = writeln!(text, "!llvm.module.flags = !{{{}}}", refs(&flags));
    let _ = writeln!(text, "!{list} = !{{!{entry}}}");
    let _ = writeln!(text, "!air.compile_options = !{{{}}}", refs(&options));
    let _ = writeln!(text, "!air.version = !{{!{version}}}");
    let _ = writeln!(text, "!air.language_version = !{{!{language}}}");
    text.push('\n');
    for (id, node) in md.nodes.iter().enumerate() {
        let _ = writeln!(text, "!{id} = {node}");
    }
    text
}

/// Numbered metadata nodes in definition order.
#[derive(Default)]
struct Metadata {
    nodes: Vec<String>,
}

impl Metadata {
    fn node(&mut self, text: String) -> usize {
        self.nodes.push(text);
        self.nodes.len() - 1
    }
}

fn refs(ids: &[usize]) -> String {
    ids.iter()
        .map(|id| format!("!{id}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn list_of(ids: &[usize]) -> String {
    format!("!{{{}}}", refs(ids))
}

/// One function: its signature, body lines, the declarations it calls, and its interface
/// metadata nodes (outputs in return-value order, inputs in parameter order).
struct Function {
    ret: String,
    params: Vec<String>,
    param_names: Vec<&'static str>,
    body: Vec<String>,
    declarations: Vec<&'static str>,
    outputs: Vec<String>,
    inputs: Vec<String>,
}

/// An LLVM metadata string of `bytes`: printable ASCII other than `"` and `\` as itself, every
/// other byte as `\XX`.
fn md_string(bytes: &[u8]) -> String {
    let mut out = String::from("!\"");
    for &byte in bytes {
        if byte.is_ascii_graphic() && byte != b'"' && byte != b'\\' || byte == b' ' {
            out.push(char::from(byte));
        } else {
            let _ = write!(out, "\\{byte:02X}");
        }
    }
    out.push('"');
    out
}

fn key_string(key: &UserKey) -> String {
    md_string(&key.0)
}

/// The IR spelling of a scalar.
fn scalar_ir(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::F32 => "float",
        Scalar::F16 => "half",
        Scalar::I32 => "i32",
        Scalar::I16 => "i16",
        Scalar::Bool => "i1",
    }
}

fn type_ir(ty: VaryingType) -> String {
    if ty.lanes == 1 {
        scalar_ir(ty.scalar).into()
    } else {
        format!("<{} x {}>", ty.lanes, scalar_ir(ty.scalar))
    }
}

/// A float constant of `scalar` (`F32` or `F16`) equal to `value`, one of the exact values the
/// companions use.
fn float_constant(scalar: Scalar, value: f32) -> String {
    match scalar {
        Scalar::F16 => {
            let bits: u16 = match value {
                0.0 => 0x0000,
                0.25 => 0x3400,
                0.5 => 0x3800,
                1.0 => 0x3C00,
                _ => unreachable!("no companion writes the half {value}"),
            };
            format!("half 0xH{bits:04X}")
        }
        _ => format!("float {value:.6e}"),
    }
}

/// The typed constant of `scalar` whose bits at the scalar's width are `bits` (an authored
/// varying value): a float in LLVM's hexadecimal spelling, the double of the same value (for a NaN,
/// of the same sign and the payload shifted to the double's top bits, the spelling LLVM prints), a
/// half as its binary16 pattern, an integer as its signed value.
fn bits_constant(scalar: Scalar, bits: u32) -> String {
    match scalar {
        Scalar::F32 => {
            let value = f32::from_bits(bits);
            let double = if value.is_nan() {
                (u64::from(bits >> 31) << 63)
                    | (0x7FF << 52)
                    | (u64::from(bits & 0x007F_FFFF) << 29)
            } else {
                f64::from(value).to_bits()
            };
            format!("float 0x{double:016X}")
        }
        Scalar::F16 => format!("half 0xH{bits:04X}"),
        Scalar::I32 => format!("i32 {}", bits as i32),
        Scalar::I16 => format!("i16 {}", bits as u16 as i16),
        Scalar::Bool => format!("i1 {}", bits == 1),
    }
}

/// A vector constant: `lanes` of the scalar constants `lane(i)`, or `poison` where `None`.
fn vector_constant(scalar: Scalar, lanes: &[Option<String>]) -> String {
    let ir = scalar_ir(scalar);
    let lanes = lanes
        .iter()
        .map(|lane| lane.clone().unwrap_or_else(|| format!("{ir} poison")))
        .collect::<Vec<_>>();
    format!("<{}>", lanes.join(", "))
}

fn passthrough(vertex: &PassthroughVertex) -> Function {
    let mut body = vec![
        "%ax.u = and i32 %vertex_id, 1".to_string(),
        "%ay.u = lshr i32 %vertex_id, 1".into(),
        "%ax = call float @air.convert.f.f32.u.i32(i32 %ax.u)".into(),
        "%ay = call float @air.convert.f.f32.u.i32(i32 %ay.u)".into(),
        "%px.s = fmul float %ax, 4.000000e+00".into(),
        "%px = fadd float %px.s, -1.000000e+00".into(),
        "%py.s = fmul float %ay, 4.000000e+00".into(),
        "%py = fadd float %py.s, -1.000000e+00".into(),
        "%u = fmul float %ax, 2.000000e+00".into(),
        "%v = fmul float %ay, 2.000000e+00".into(),
    ];
    let half_uv = vertex.varyings.iter().any(|varying| {
        varying.ty.scalar == Scalar::F16
            && matches!(
                varying.value,
                PassthroughValue::Uv | PassthroughValue::UvZ { .. } | PassthroughValue::UvHalfOne
            )
    });
    if half_uv {
        body.push("%u.h = fptrunc float %u to half".into());
        body.push("%v.h = fptrunc float %v to half".into());
    }
    body.push(format!(
        "%position.x = insertelement <4 x float> {}, float %px, i32 0",
        vector_constant(
            Scalar::F32,
            &[
                None,
                None,
                Some(float_constant(Scalar::F32, 0.0)),
                Some(float_constant(Scalar::F32, 1.0)),
            ]
        )
    ));
    body.push("%position = insertelement <4 x float> %position.x, float %py, i32 1".into());

    let mut fields = vec![("<4 x float>".to_string(), "%position".to_string())];
    let mut outputs = vec![
        "!{!\"air.position\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", \
         !\"metal2vulkan_position\"}"
            .to_string(),
    ];
    for (present, role, name) in [
        (
            vertex.viewport_array_index,
            "air.viewport_array_index",
            "metal2vulkan_viewport",
        ),
        (
            vertex.render_target_array_index,
            "air.render_target_array_index",
            "metal2vulkan_layer",
        ),
    ] {
        if present {
            fields.push(("i32".into(), "0".into()));
            outputs.push(format!(
                "!{{!\"{role}\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", \
                 !\"{name}\"}}"
            ));
        }
    }
    for (index, varying) in vertex.varyings.iter().enumerate() {
        let value = passthrough_value(&mut body, index, varying.ty, varying.value);
        fields.push((type_ir(varying.ty), value));
        outputs.push(format!(
            "!{{!\"air.vertex_output\", {}, !\"air.arg_type_name\", {}, !\"air.arg_name\", \
             !\"metal2vulkan_varying_{index}\"}}",
            key_string(&varying.key),
            md_string(varying.type_name.as_bytes()),
        ));
    }
    let ret = format!(
        "<{{ {} }}>",
        fields
            .iter()
            .map(|(ty, _)| ty.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut aggregate = "undef".to_string();
    for (index, (ty, value)) in fields.iter().enumerate() {
        body.push(format!(
            "%out.{index} = insertvalue {ret} {aggregate}, {ty} {value}, {index}"
        ));
        aggregate = format!("%out.{index}");
    }
    body.push(format!("ret {ret} {aggregate}"));
    Function {
        ret,
        params: vec!["i32".into()],
        param_names: vec!["vertex_id"],
        body,
        declarations: vec!["declare float @air.convert.f.f32.u.i32(i32)"],
        outputs,
        inputs: vec![
            "!{i32 0, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", \
             !\"vertex_id\"}"
                .into(),
        ],
    }
}

/// Append the instructions of varying `index`'s value to `body`; the operand that holds it (an
/// SSA name or a constant).
fn passthrough_value(
    body: &mut Vec<String>,
    index: usize,
    ty: VaryingType,
    value: PassthroughValue,
) -> String {
    let scalar = ty.scalar;
    let lanes = ty.lanes as usize;
    let integer = |lane: usize| match scalar {
        Scalar::Bool => format!("i1 {}", lane.is_multiple_of(2)),
        _ => format!("{} {}", scalar_ir(scalar), lane + 1),
    };
    let constant = |lane_values: Vec<String>| {
        if lanes == 1 {
            lane_values[0]
                .split_once(' ')
                .map(|(_, value)| value.to_string())
                .unwrap_or_default()
        } else {
            vector_constant(
                scalar,
                &lane_values.into_iter().map(Some).collect::<Vec<_>>(),
            )
        }
    };
    let tail: &[f32] = match value {
        PassthroughValue::Counting | PassthroughValue::Alternating => {
            return constant((0..lanes).map(integer).collect());
        }
        PassthroughValue::Quarter => return constant(vec![float_constant(scalar, 0.25)]),
        PassthroughValue::Authored { lanes: bits } => {
            return constant(
                bits[..lanes]
                    .iter()
                    .map(|&bits| bits_constant(scalar, bits))
                    .collect(),
            );
        }
        PassthroughValue::Uv => &[],
        PassthroughValue::UvZ { z_one: false } => &[0.5],
        PassthroughValue::UvZ { z_one: true } => &[1.0],
        PassthroughValue::UvHalfOne => &[0.5, 1.0],
    };
    let (u, v) = match scalar {
        Scalar::F16 => ("%u.h", "%v.h"),
        _ => ("%u", "%v"),
    };
    let vector = type_ir(ty);
    let ir = scalar_ir(scalar);
    let start = vector_constant(
        scalar,
        &[None, None]
            .into_iter()
            .chain(
                tail.iter()
                    .map(|&value| Some(float_constant(scalar, value))),
            )
            .collect::<Vec<_>>(),
    );
    body.push(format!(
        "%varying.{index}.u = insertelement {vector} {start}, {ir} {u}, i32 0"
    ));
    body.push(format!(
        "%varying.{index} = insertelement {vector} %varying.{index}.u, {ir} {v}, i32 1"
    ));
    format!("%varying.{index}")
}

fn observer(fragment: &ObserverFragment) -> Function {
    let (element, type_name) = match fragment.output {
        ObserverOutput::Float => ("float", "float4"),
        ObserverOutput::Sint => ("i32", "int4"),
        ObserverOutput::Uint => ("i32", "uint4"),
    };
    let ret = format!("<4 x {element}>");
    let outputs = vec![format!(
        "!{{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"{type_name}\"}}"
    )];
    match &fragment.input {
        ObservedValue::Position => Function {
            ret: ret.clone(),
            params: vec!["<4 x float>".into()],
            param_names: vec!["observed"],
            body: vec![format!("ret {ret} %observed")],
            declarations: Vec::new(),
            outputs,
            inputs: vec![
                "!{i32 0, !\"air.position\", !\"air.center\", !\"air.no_perspective\", \
                 !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"observed\"}"
                    .into(),
            ],
        },
        ObservedValue::Varying {
            key,
            ty,
            type_name: input_type,
        } => {
            let interpolation = if ty.scalar.is_float() {
                "!\"air.center\", !\"air.perspective\""
            } else {
                "!\"air.flat\""
            };
            let mut body = Vec::new();
            let pad = |lane: usize| {
                let value = if lane == 3 { 1 } else { 0 };
                match fragment.output {
                    ObserverOutput::Float => float_constant(Scalar::F32, value as f32),
                    ObserverOutput::Sint | ObserverOutput::Uint => format!("i32 {value}"),
                }
            };
            let lanes = ty.lanes as usize;
            let mut current = format!(
                "<{}>",
                (0..4)
                    .map(|lane| if lane < lanes {
                        format!("{element} poison")
                    } else {
                        pad(lane)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let source = type_ir(*ty);
            let scalar = scalar_ir(ty.scalar);
            for lane in 0..lanes {
                let value = if lanes == 1 {
                    "%observed".to_string()
                } else {
                    body.push(format!(
                        "%lane.{lane} = extractelement {source} %observed, i32 {lane}"
                    ));
                    format!("%lane.{lane}")
                };
                let widened = match (ty.scalar, fragment.output) {
                    (Scalar::F32 | Scalar::I32, _) => value,
                    (Scalar::F16, _) => {
                        body.push(format!("%wide.{lane} = fpext half {value} to float"));
                        format!("%wide.{lane}")
                    }
                    (Scalar::I16, ObserverOutput::Sint) => {
                        body.push(format!("%wide.{lane} = sext i16 {value} to i32"));
                        format!("%wide.{lane}")
                    }
                    (Scalar::I16 | Scalar::Bool, _) => {
                        body.push(format!("%wide.{lane} = zext {scalar} {value} to i32"));
                        format!("%wide.{lane}")
                    }
                };
                body.push(format!(
                    "%color.{lane} = insertelement {ret} {current}, {element} {widened}, i32 {lane}"
                ));
                current = format!("%color.{lane}");
            }
            body.push(format!("ret {ret} {current}"));
            Function {
                ret,
                params: vec![source],
                param_names: vec!["observed"],
                body,
                declarations: Vec::new(),
                outputs,
                inputs: vec![format!(
                    "!{{i32 0, !\"air.fragment_input\", {}, {interpolation}, \
                     !\"air.arg_type_name\", {}, !\"air.arg_name\", !\"observed\"}}",
                    key_string(key),
                    md_string(input_type.as_bytes()),
                )],
            }
        }
    }
}
