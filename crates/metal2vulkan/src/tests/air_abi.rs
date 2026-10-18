//! The AIR metadata ABI over the committed public fixtures.
//!
//! Every public `.ll` fixture is assembled with `llvm-as` at test time (opaque pointers) and every
//! `engine/*.air.ll` companion with `xcrun metal-as` (typed pointers, as Apple's compiler
//! writes them). The tests check that each fixture's ABI decodes and cross-checks against its IR,
//! that each entry kind decodes to the roles its metadata states, and that the typed and opaque
//! spellings of one fixture decode to the same ABI (pointees only feed decode-time checks). A
//! missing tool skips its test visibly; a tool that rejects any fixture fails it.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::air::abi::{
    self, AbiInt, Access, AddressMode, AirAbi, Builtin, DepthQualifier, EntryPoint, Filter,
    MeshField, MeshFieldRole, MeshOutputs, MeshTopology, OutputRole, ParamRole, PatchInput, Stage,
    VisibleTarget,
};
use crate::air::bitcode::decode;
use crate::air::ir::{FuncId, Type, TypeId, TypeTable};
use crate::budget::{Budget, Meter};
use crate::pipeline::testing::Scratch;

// ---- Tools and fixtures ----

/// `llvm-as`: the `METAL2VULKAN_LLVM_AS` override, else `PATH`, else the Homebrew LLVM prefixes.
fn llvm_as() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("METAL2VULKAN_LLVM_AS") {
        return Some(PathBuf::from(p));
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .chain(
            ["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"]
                .iter()
                .map(PathBuf::from),
        )
        .map(|d| d.join("llvm-as"))
        .find(|p| p.is_file())
}

fn have_metal_as() -> bool {
    cfg!(target_os = "macos")
        && Command::new("xcrun")
            .args(["-f", "metal-as"])
            .output()
            .is_ok_and(|o| o.status.success())
}

fn assemble(tool: &Path, args: &[&std::ffi::OsStr], out: &Path) -> Option<Vec<u8>> {
    let status = Command::new(tool)
        .args(args)
        .arg("-o")
        .arg(out)
        .output()
        .ok()?;
    if !status.status.success() {
        return None;
    }
    std::fs::read(out).ok()
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/fixtures/public")
}

fn engine_dir() -> PathBuf {
    fixtures_dir().join("engine")
}

/// Every committed opaque-pointer `.ll` fixture, sorted.
fn opaque_fixtures() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in [fixtures_dir(), engine_dir()] {
        out.extend(
            std::fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().path())
                .filter(|p| {
                    let name = p.file_name().unwrap().to_string_lossy();
                    name.ends_with(".ll") && !name.ends_with(".air.ll")
                }),
        );
    }
    out.sort();
    out
}

/// The bitcode of `ll`. Every committed fixture assembles, so a rejection fails.
fn llvm_assemble(tool: &Path, ll: &Path, scratch: &Scratch) -> Vec<u8> {
    let out = scratch.path().join("fixture.bc");
    assemble(tool, &[ll.as_os_str()], &out).unwrap_or_else(|| panic!("llvm-as rejected {ll:?}"))
}

fn metal_assemble(air_ll: &Path, scratch: &Scratch) -> Vec<u8> {
    let out = scratch.path().join("fixture.air");
    let args = [std::ffi::OsStr::new("metal-as"), air_ll.as_os_str()];
    assemble(Path::new("xcrun"), &args, &out)
        .unwrap_or_else(|| panic!("xcrun metal-as rejected {air_ll:?}"))
}

/// Decode `bytes`; the ABI is decoded with the module.
fn abi_of(bytes: &[u8], what: &Path) -> AirAbi {
    let mut m = Meter::new(&Budget::default());
    decode(bytes, &mut m)
        .unwrap_or_else(|e| panic!("{what:?}: {e}"))
        .abi
}

/// The ABI of one spelling of a fixture, with what its module-local ids denote resolved.
struct Spelling {
    abi: AirAbi,
    /// Every function constant's value type: a `TypeId` indexes its own module's type table, and
    /// the typed-pointer spelling interns more (pointer) types, so the ids of one type differ
    /// between spellings while the types agree.
    constant_types: Vec<String>,
    /// Every function's name, indexed by its `FuncId`: a `FuncId` indexes its own module's
    /// function list, whose order the two spellings need not share (see
    /// `typed_and_opaque_spellings_decode_the_same_abi`).
    functions: Vec<Vec<u8>>,
}

fn spelling(bytes: &[u8], what: &Path) -> Spelling {
    let mut m = Meter::new(&Budget::default());
    let d = decode(bytes, &mut m).unwrap_or_else(|e| panic!("{what:?}: {e}"));
    let constant_types = d
        .abi
        .function_constants
        .iter()
        .map(|c| spelled(d.module.types(), c.ty))
        .collect();
    let functions = d
        .module
        .parts()
        .funcs
        .values()
        .map(|f| {
            let name = d.module.symbols().get(f.name);
            name.unwrap_or_else(|| panic!("{what:?}: unnamed function"))
                .to_vec()
        })
        .collect();
    Spelling {
        abi: d.abi,
        constant_types,
        functions,
    }
}

/// `abi` with every function id it holds replaced by `to` of it.
fn renumber_functions(abi: &mut AirAbi, to: impl Fn(FuncId) -> FuncId) {
    for ep in &mut abi.entries {
        ep.func = to(ep.func);
        for p in &mut ep.params {
            if let ParamRole::PatchInput(PatchInput::ControlPoints { function, .. }) = &mut p.role {
                *function = to(*function);
            }
        }
    }
    for r in &mut abi.visible_function_refs {
        if let VisibleTarget::Function(f) = &mut r.target {
            *f = to(*f);
        }
    }
    for v in &mut abi.visible_functions {
        v.func = to(v.func);
        if let Some(s) = &mut v.stitching {
            let types = s
                .result
                .iter_mut()
                .chain(s.arguments.iter_mut().map(|a| &mut a.ty));
            for f in types.flat_map(|t| t.helpers.iter_mut().flatten()) {
                *f = to(*f);
            }
        }
    }
    for k in &mut abi.core_image_kernels {
        k.func = to(k.func);
    }
}

/// `id` with every type it refers to spelled out in place of its (module-local) id, so a vector
/// or aggregate constant type compares across the two spellings too.
fn spelled(types: &TypeTable, id: TypeId) -> String {
    let ty = types.get(id).expect("constant type");
    let each = |ids: &[TypeId]| {
        ids.iter()
            .map(|&child| spelled(types, child))
            .collect::<Vec<_>>()
            .join(", ")
    };
    match ty {
        Type::Vector { elem, len } => format!("<{len} x {}>", spelled(types, *elem)),
        Type::Array { elem, len } => format!("[{len} x {}]", spelled(types, *elem)),
        Type::Struct {
            fields,
            packed,
            name,
        } => format!("struct {name:?} packed={packed} {{{}}}", each(fields)),
        Type::Function {
            ret,
            params,
            vararg,
        } => format!(
            "fn({}) -> {} vararg={vararg}",
            each(params),
            spelled(types, *ret)
        ),
        leaf => {
            assert!(leaf.children().is_empty(), "{leaf:?} refers to other types");
            format!("{leaf:?}")
        }
    }
}

/// The ABI of the opaque fixture `name` (relative to the public fixture directory).
fn fixture_abi(tool: &Path, name: &str) -> AirAbi {
    let scratch = Scratch::new("m2v-air-abi");
    let path = fixtures_dir().join(name);
    let bytes = llvm_assemble(tool, &path, &scratch);
    abi_of(&bytes, &path)
}

fn slot_index(r: &ParamRole) -> Option<AbiInt> {
    match r {
        ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) | ParamRole::Threadgroup(b) => {
            Some(b.location.index)
        }
        ParamRole::Texture(t) => Some(t.location.index),
        ParamRole::Sampler { location }
        | ParamRole::VertexInput { location }
        | ParamRole::StageIn { location } => Some(location.index),
        _ => None,
    }
}

// ---- Tests ----

/// Every public fixture assembles and its ABI decodes and agrees with its IR, and together they
/// cover every entry kind, function constants, and constexpr samplers. No fixture is excused: a
/// hand-written one writes its metadata in Apple's ABI grammar too (`air::abi`'s own test
/// `a_nested_layout_named_after_its_member_is_refused` pins the refusal of a nested layout spelled
/// otherwise).
#[test]
fn every_public_fixture_decodes_its_abi() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP: llvm-as not found; set METAL2VULKAN_LLVM_AS");
        return;
    };
    let scratch = Scratch::new("m2v-air-abi");
    let (mut kernels, mut vertex, mut fragment, mut meshes) = (0, 0, 0, 0);
    let (mut constants, mut samplers) = (0, 0);
    let mut decoded = 0;
    for path in opaque_fixtures() {
        let bytes = llvm_assemble(&tool, &path, &scratch);
        let mut m = Meter::new(&Budget::default());
        // Relative to the public fixture directory: a `engine/` fixture may share a top-level
        // fixture's file name (the compiled replacements do).
        let name = path
            .strip_prefix(fixtures_dir())
            .unwrap()
            .to_string_lossy()
            .into_owned();
        // `bitcode::decode` decodes the ABI with the module, so a malformed ABI refuses the
        // module; the result is what `abi::decode` returns for the same module.
        let d = decode(&bytes, &mut m).unwrap_or_else(|e| panic!("{name}: {e}"));
        let abi = abi::decode(&d.module, &d.pointees, &mut m).unwrap();
        assert_eq!(abi, d.abi, "{name}");
        decoded += 1;
        for e in &abi.entries {
            match e.stage {
                Stage::Kernel { .. } => kernels += 1,
                Stage::Vertex { .. } => vertex += 1,
                Stage::Fragment { .. } => fragment += 1,
                Stage::Object { .. } | Stage::Mesh { .. } => meshes += 1,
            }
        }
        constants += abi.function_constants.len();
        samplers += abi.static_samplers.len();
    }
    assert!(decoded > 100, "only {decoded} fixtures decoded");
    assert!(
        kernels > 0 && vertex > 0 && fragment > 0 && meshes > 0,
        "{kernels} {vertex} {fragment} {meshes}"
    );
    assert!(constants > 0 && samplers > 0, "{constants} {samplers}");
}

/// The entry of `abi` named `name`.
fn entry<'a>(abi: &'a AirAbi, name: &str) -> &'a EntryPoint {
    abi.entries
        .iter()
        .find(|e| e.name.to_string() == name)
        .unwrap_or_else(|| panic!("no entry {name}"))
}

/// The mesh output type of mesh function `e`.
fn mesh_outputs(e: &EntryPoint) -> &MeshOutputs {
    e.params
        .iter()
        .find_map(|p| match &p.role {
            ParamRole::Mesh(m) => Some(&**m),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{}: no mesh output", e.name))
}

/// Apple's object and mesh functions decode: their stages and stage attributes, the payload, the
/// grid handle, and the mesh output type's fields, counts and topology.
#[test]
fn object_and_mesh_roles() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP: llvm-as not found; set METAL2VULKAN_LLVM_AS");
        return;
    };
    let abi = fixture_abi(&tool, "engine/mesh_order.ll");
    let object = entry(&abi, "mesh_order_object");
    assert_eq!(
        object.stage,
        Stage::Object {
            max_total_threads: None,
            max_mesh_threadgroups: None
        }
    );
    let roles: Vec<_> = object.params.iter().map(|p| &p.role).collect();
    assert!(
        matches!(roles[0], ParamRole::Payload { layout: Some(_) }),
        "{roles:?}"
    );
    assert_eq!(roles[1], &ParamRole::MeshGridProperties);
    assert_eq!(
        roles[2],
        &ParamRole::Builtin(Builtin::ThreadIndexInThreadgroup)
    );
    let mesh = entry(&abi, "mesh_order_mesh");
    assert_eq!(
        mesh.stage,
        Stage::Mesh {
            max_total_threads: None
        }
    );
    assert!(mesh
        .params
        .iter()
        .any(|p| matches!(p.role, ParamRole::Payload { layout: Some(_) })));
    let out = mesh_outputs(mesh);
    assert_eq!(
        (out.max_vertices, out.max_primitives, out.topology),
        (4, 4, MeshTopology::Triangle)
    );
    let vertex: Vec<_> = out.vertex.iter().map(|f| &f.role).collect();
    assert!(
        matches!(
            vertex[..],
            [
                MeshFieldRole::Position { invariant: false },
                MeshFieldRole::VertexData { slot: 0, .. }
            ]
        ),
        "{vertex:?}"
    );
    let primitive: Vec<_> = out.primitive.iter().map(|f| &f.role).collect();
    assert!(
        matches!(
            primitive[..],
            [
                MeshFieldRole::PrimitiveData { slot: 0, .. },
                MeshFieldRole::PrimitiveData { slot: 1, .. },
                MeshFieldRole::PrimitiveCulled
            ]
        ),
        "{primitive:?}"
    );
    // The same mesh without an object stage takes no payload.
    let solo = entry(&abi, "mesh_order_solo");
    assert!(!solo
        .params
        .iter()
        .any(|p| matches!(p.role, ParamRole::Payload { .. })));
    assert!(matches!(
        entry(&abi, "mesh_order_fragment").stage,
        Stage::Fragment { .. }
    ));

    let abi = fixture_abi(&tool, "engine/mesh_outputs.ll");
    let points = mesh_outputs(entry(&abi, "mesh_outputs_points"));
    assert_eq!(
        (points.max_vertices, points.max_primitives, points.topology),
        (16, 8, MeshTopology::Point)
    );
    let vertex: Vec<_> = points.vertex.iter().map(|f| &f.role).collect();
    assert!(
        matches!(
            vertex[..],
            [
                MeshFieldRole::Position { invariant: false },
                MeshFieldRole::PointSize,
                MeshFieldRole::ClipDistance {
                    array_size: Some(2)
                },
                MeshFieldRole::VertexData { slot: 0, .. },
                MeshFieldRole::VertexData { slot: 1, .. },
                MeshFieldRole::VertexData { slot: 2, .. }
            ]
        ),
        "{vertex:?}"
    );
    let primitive: Vec<_> = points.primitive.iter().map(|f| &f.role).collect();
    assert!(
        matches!(
            primitive[..],
            [
                MeshFieldRole::PrimitiveCulled,
                MeshFieldRole::PrimitiveId,
                MeshFieldRole::RenderTargetArrayIndex,
                MeshFieldRole::ViewportArrayIndex,
                MeshFieldRole::PrimitiveData { slot: 0, .. }
            ]
        ),
        "{primitive:?}"
    );
    let lines_entry = entry(&abi, "mesh_outputs_lines");
    assert_eq!(
        lines_entry.stage,
        Stage::Mesh {
            max_total_threads: Some(32)
        }
    );
    let lines = mesh_outputs(lines_entry);
    assert_eq!(
        (lines.max_vertices, lines.max_primitives, lines.topology),
        (8, 4, MeshTopology::Line)
    );
    assert!(matches!(
        lines.vertex[..],
        [MeshField {
            role: MeshFieldRole::Position { invariant: true },
            ..
        }]
    ));
    assert!(lines.primitive.is_empty());
    let object = entry(&abi, "mesh_outputs_object");
    assert_eq!(
        object.stage,
        Stage::Object {
            max_total_threads: Some(64),
            max_mesh_threadgroups: Some(16)
        }
    );
    assert!(object
        .params
        .iter()
        .any(|p| matches!(p.role, ParamRole::Threadgroup(_))));
}

#[test]
fn kernel_roles() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP: llvm-as not found; set METAL2VULKAN_LLVM_AS");
        return;
    };
    let abi = fixture_abi(&tool, "engine/storage_texture_write.ll");
    let [e] = &abi.entries[..] else {
        panic!("one entry")
    };
    assert!(matches!(e.stage, Stage::Kernel { .. }));
    let ParamRole::Texture(t) = &e.params[0].role else {
        panic!("texture: {:?}", e.params[0].role)
    };
    assert_eq!(
        (t.location.index, t.access),
        (AbiInt::Const(0), Access::Write)
    );
    assert_eq!(
        e.params[1].role,
        ParamRole::Builtin(Builtin::ThreadPositionInGrid)
    );
    let abi = fixture_abi(&tool, "engine/buffer_end_byte_load_constant.ll");
    let spaces: Vec<_> = abi.entries[0]
        .params
        .iter()
        .filter_map(|p| match &p.role {
            ParamRole::Buffer(b) => Some((slot_index(&p.role), b.space.raw(), p.ty.size)),
            _ => None,
        })
        .collect();
    assert!(
        spaces.contains(&(Some(AbiInt::Const(0)), 2, Some(6))),
        "{spaces:?}"
    );
}

/// The vertex output and the fragment input of one varying carry the same user key.
#[test]
fn vertex_and_fragment_roles() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP: llvm-as not found; set METAL2VULKAN_LLVM_AS");
        return;
    };
    let vs = fixture_abi(&tool, "vertex_varying_triangle.ll");
    let vs = &vs.entries[0];
    assert!(matches!(vs.stage, Stage::Vertex { patch: None }));
    let inputs: Vec<_> = vs.params.iter().map(|p| slot_index(&p.role)).collect();
    assert_eq!(inputs, vec![Some(AbiInt::Const(0)), Some(AbiInt::Const(1))]);
    let OutputRole::VertexOutput(out_key) = &vs.outputs[1].role else {
        panic!("vertex output")
    };
    assert_eq!(
        vs.outputs[0].role,
        OutputRole::Position { invariant: false }
    );

    let fs = fixture_abi(&tool, "fragment_varying_color.ll");
    let fs = &fs.entries[0];
    let ParamRole::FragmentInput(v) = &fs.params[1].role else {
        panic!("fragment input")
    };
    assert_eq!(&v.key, out_key);
    assert!(fs.params[0].unused);
    assert!(matches!(
        fs.outputs[0].role,
        OutputRole::RenderTarget {
            attachment: AbiInt::Const(0),
            index: 0,
            ..
        }
    ));

    let ds = fixture_abi(&tool, "fragment_depth_stencil.ll");
    let roles: Vec<_> = ds.entries[0]
        .outputs
        .iter()
        .map(|o| o.role.clone())
        .collect();
    assert_eq!(
        roles,
        vec![OutputRole::Depth(DepthQualifier::Any), OutputRole::Stencil]
    );
}

#[test]
fn function_constants_and_constexpr_samplers() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP: llvm-as not found; set METAL2VULKAN_LLVM_AS");
        return;
    };
    let abi = fixture_abi(&tool, "kernel_function_constant_gate_fold.ll");
    let [fc] = &abi.function_constants[..] else {
        panic!("one function constant")
    };
    assert_eq!(fc.index, 0);

    let abi = fixture_abi(&tool, "kernel_bicubic_repeat_wrap.ll");
    let [s] = &abi.static_samplers[..] else {
        panic!("one sampler")
    };
    assert_eq!(s.state.mag_filter, Filter::Bicubic);
    assert_eq!(s.state.address_s, AddressMode::Repeat);
}

/// The typed-pointer and opaque-pointer spellings of a fixture decode to the same ABI; the typed
/// spelling additionally passes the pointee checks.
///
/// Module-local ids are compared through what they denote: a constant's type by its spelling, a
/// function by its name. The two spellings are different bitcode modules, and their function lists
/// need not share an order: LLVM's bitcode reader upgrades a typed-pointer intrinsic declaration
/// such as `llvm.lifetime.start.p0i8` to its opaque-pointer name (`llvm.lifetime.start.p0`) by
/// adding the new declaration at the end of the module, so the `.ll` that `llvm-dis` writes lists
/// it last, while `metal-as` keeps it where Apple's compiler put it, between the definitions.
#[test]
fn typed_and_opaque_spellings_decode_the_same_abi() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP: llvm-as not found; set METAL2VULKAN_LLVM_AS");
        return;
    };
    if !have_metal_as() {
        eprintln!("SKIP: xcrun metal-as not available (typed-pointer bitcode)");
        return;
    }
    let mut compared = 0;
    for entry in std::fs::read_dir(engine_dir()).unwrap() {
        let typed = entry.unwrap().path();
        let name = typed.file_name().unwrap().to_string_lossy().into_owned();
        let Some(stem) = name.strip_suffix(".air.ll") else {
            continue;
        };
        let scratch = Scratch::new("m2v-air-abi");
        let mut typed_spelling = spelling(&metal_assemble(&typed, &scratch), &typed);
        let opaque = engine_dir().join(format!("{stem}.ll"));
        let bytes = llvm_assemble(&tool, &opaque, &scratch);
        let opaque_spelling = spelling(&bytes, &opaque);
        assert_eq!(
            typed_spelling.constant_types, opaque_spelling.constant_types,
            "{stem}: function constant types"
        );
        for (t, o) in typed_spelling
            .abi
            .function_constants
            .iter_mut()
            .zip(&opaque_spelling.abi.function_constants)
        {
            t.ty = o.ty;
        }
        let typed_functions = &typed_spelling.functions;
        let opaque_functions = &opaque_spelling.functions;
        renumber_functions(&mut typed_spelling.abi, |f| {
            let name = &typed_functions[f.raw() as usize];
            let at = opaque_functions.iter().position(|o| o == name);
            let at = at.unwrap_or_else(|| {
                let name = String::from_utf8_lossy(name);
                panic!("{stem}: the opaque spelling has no function {name}")
            });
            FuncId::new(u32::try_from(at).unwrap())
        });
        assert_eq!(typed_spelling.abi, opaque_spelling.abi, "{stem}");
        compared += 1;
    }
    assert!(compared > 0);
}
