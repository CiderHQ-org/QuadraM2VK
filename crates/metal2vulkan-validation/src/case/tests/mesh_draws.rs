use super::*;

/// A mesh case of `example()`'s shape: its entry a mesh function drawn with a fragment partner,
/// three threadgroups of four threads, into one render target it selects.
fn mesh_case() -> AuthoredCase {
    let mut case = example();
    case.case_id = "22".repeat(32);
    case.stage = Stage::Mesh;
    case.entry = "mesh_main".into();
    case.buffers.clear();
    case.dispatch = None;
    case.mesh_draw = Some(MeshDraw {
        object: None,
        mesh: None,
        fragment: Some("fragment_main".into()),
        grid: MeshGrid::Threadgroups([3, 1, 1]),
        threads_per_object_threadgroup: [1, 1, 1],
        threads_per_mesh_threadgroup: [4, 1, 1],
        rasterization: true,
        observation: None,
        object_launch: None,
    });
    case.render_targets = vec![RenderTargetResource {
        index: 0,
        format: TextureFormat::Rgba32Uint,
        dimensions: [8, 8],
        initial_bytes_b64: base64::engine::general_purpose::STANDARD.encode([0u8; 8 * 8 * 16]),
    }];
    case.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [8, 8],
    };
    case
}

fn errors(case: &AuthoredCase) -> Vec<String> {
    case.validate_literal_resources().err().unwrap_or_default()
}

fn refused(case: &AuthoredCase, what: &str) {
    let errors = errors(case);
    assert!(
        errors.iter().any(|error| error.contains(what)),
        "{what:?} not among {errors:?}"
    );
}

/// The draw is spelled as Metal's two calls name their grids, rasterization is written only when
/// off, and an absent draw leaves every other case's identity as it was.
#[test]
fn a_mesh_draw_round_trips_in_its_one_spelling() {
    let case = mesh_case();
    let json = serde_json::to_value(&case).unwrap();
    assert_eq!(
        json["mesh_draw"],
        serde_json::json!({
            "fragment": "fragment_main",
            "grid": {"threadgroups": [3, 1, 1]},
            "threads_per_object_threadgroup": [1, 1, 1],
            "threads_per_mesh_threadgroup": [4, 1, 1],
        })
    );
    assert_eq!(serde_json::from_value::<AuthoredCase>(json).unwrap(), case);
    let mut threads = case.clone();
    let draw = threads.mesh_draw.as_mut().unwrap();
    draw.grid = MeshGrid::Threads([12, 1, 1]);
    draw.rasterization = false;
    draw.fragment = None;
    let json = serde_json::to_value(&threads).unwrap();
    assert_eq!(
        json["mesh_draw"]["grid"],
        serde_json::json!({"threads": [12, 1, 1]})
    );
    assert_eq!(json["mesh_draw"]["rasterization"], serde_json::json!(false));
    assert_ne!(
        case.computed_case_id().unwrap(),
        threads.computed_case_id().unwrap()
    );
    assert!(serde_json::to_value(example())
        .unwrap()
        .get("mesh_draw")
        .is_none());
}

#[test]
fn a_mesh_case_with_its_draw_is_a_valid_literal() {
    assert_eq!(errors(&mesh_case()), Vec::<String>::new());
    let mut object = mesh_case();
    object.stage = Stage::Object;
    object.entry = "object_main".into();
    let draw = object.mesh_draw.as_mut().unwrap();
    draw.mesh = Some("mesh_main".into());
    draw.grid = MeshGrid::Threads([6, 1, 1]);
    object.threadgroup_memory = vec![ThreadgroupMemoryResource {
        binding: 0,
        length: 16,
    }];
    assert_eq!(errors(&object), Vec::<String>::new());
}

#[test]
fn a_mesh_draw_is_an_object_or_mesh_case_s_only_work() {
    let mut kernel = example();
    kernel.dispatch = None;
    kernel.mesh_draw = mesh_case().mesh_draw;
    refused(&kernel, "mesh_draw is valid only for object and mesh cases");
    let mut both = mesh_case();
    both.dispatch = example().dispatch;
    refused(
        &both,
        "exactly one of dispatch, draw, tessellation, or mesh_draw",
    );
    let mut none = mesh_case();
    none.mesh_draw = None;
    refused(&none, "object or mesh case requires mesh_draw");
}

#[test]
fn a_mesh_draw_names_its_partners_and_not_its_own_entry() {
    let mut own = mesh_case();
    own.mesh_draw.as_mut().unwrap().mesh = Some("other_mesh".into());
    refused(&own, "mesh_draw.mesh of a mesh case");
    let mut object = mesh_case();
    object.stage = Stage::Object;
    object.mesh_draw.as_mut().unwrap().object = Some("object_main".into());
    refused(&object, "mesh_draw.object of an object case");
    let mut itself = mesh_case();
    itself.mesh_draw.as_mut().unwrap().fragment = Some("mesh_main".into());
    refused(&itself, "names the case's own entry");
    let mut empty = mesh_case();
    empty.mesh_draw.as_mut().unwrap().fragment = Some(" ".into());
    refused(&empty, "mesh_draw.fragment must not be empty");
}

/// Metal's limits: an axis of zero is no grid this schema states, and a threadgroup of more than
/// 1024 threads is one Metal's validation stops.
#[test]
fn a_mesh_draw_s_grid_and_threadgroups_are_metal_s() {
    let mut zero = mesh_case();
    zero.mesh_draw.as_mut().unwrap().grid = MeshGrid::Threadgroups([3, 0, 1]);
    refused(&zero, "mesh_draw.grid");
    let mut wide = mesh_case();
    wide.mesh_draw
        .as_mut()
        .unwrap()
        .threads_per_mesh_threadgroup = [32, 32, 2];
    refused(&wide, "holds 2048 threads");
    let mut object = mesh_case();
    object
        .mesh_draw
        .as_mut()
        .unwrap()
        .threads_per_object_threadgroup = [1025, 1, 1];
    refused(
        &object,
        "threads_per_object_threadgroup [1025, 1, 1] holds 1025 threads",
    );
    let mut most = mesh_case();
    most.mesh_draw
        .as_mut()
        .unwrap()
        .threads_per_mesh_threadgroup = [32, 32, 1];
    assert_eq!(errors(&most), Vec::<String>::new());
}

/// A rasterizing draw names its fragment function: Metal's descriptor validation aborts on a
/// rasterizing mesh pipeline without one (the M3 Max under macOS 27.0.1), so the schema refuses
/// the draw, and no executor ever builds that pipeline. Rasterization is on unless the case turns
/// it off.
#[test]
fn a_rasterizing_mesh_draw_names_its_fragment_function() {
    let mut unnamed = mesh_case();
    unnamed.mesh_draw.as_mut().unwrap().fragment = None;
    refused(
        &unnamed,
        "mesh_draw rasterizes and names no fragment function",
    );
    let json = serde_json::json!({
        "grid": {"threadgroups": [1, 1, 1]},
        "threads_per_object_threadgroup": [1, 1, 1],
        "threads_per_mesh_threadgroup": [1, 1, 1],
    });
    let defaulted = serde_json::from_value::<MeshDraw>(json).unwrap();
    assert!(defaulted.rasterization && defaulted.fragment.is_none());
    let mut defaulted_case = mesh_case();
    defaulted_case.mesh_draw = Some(defaulted);
    refused(
        &defaulted_case,
        "mesh_draw rasterizes and names no fragment function",
    );
}

/// A rasterizing mesh case may state an observation in its fragment partner's place, whose
/// generated observer fragment both executors draw with; never beside a fragment partner, without
/// rasterization, or in an object case, whose observer would read another function's outputs.
#[test]
fn an_observation_takes_the_fragment_partner_s_place() {
    let mut observed = mesh_case();
    let draw = observed.mesh_draw.as_mut().unwrap();
    draw.fragment = None;
    draw.observation = Some(MeshObservation::PrimitiveData { index: 0 });
    assert_eq!(errors(&observed), Vec::<String>::new());
    let json = serde_json::to_value(observed.mesh_draw.as_ref().unwrap()).unwrap();
    assert_eq!(
        json["observation"],
        serde_json::json!({"kind": "primitive_data", "index": 0})
    );

    let mut both = observed.clone();
    both.mesh_draw.as_mut().unwrap().fragment = Some("fragment_main".into());
    refused(&both, "names a fragment partner and an observation");
    let mut off = observed.clone();
    off.mesh_draw.as_mut().unwrap().rasterization = false;
    refused(&off, "mesh_draw.observation without rasterization");
    let mut object = observed;
    object.stage = Stage::Object;
    object.entry = "object_main".into();
    object.mesh_draw.as_mut().unwrap().mesh = Some("mesh_main".into());
    refused(
        &object,
        "mesh_draw.observation of an object case other than its launch",
    );
}

/// An object case may state its launch observation in its mesh and fragment partners' places,
/// over one object threadgroup of either grid kind; never beside a mesh partner, over two object
/// threadgroups (whose mesh threadgroups would draw the same rows), or in a mesh case.
#[test]
fn a_launch_observation_takes_the_mesh_partner_s_place() {
    let mut launched = mesh_case();
    launched.stage = Stage::Object;
    launched.entry = "object_main".into();
    let draw = launched.mesh_draw.as_mut().unwrap();
    draw.fragment = None;
    draw.grid = MeshGrid::Threadgroups([1, 1, 1]);
    draw.threads_per_object_threadgroup = [4, 2, 1];
    draw.observation = Some(MeshObservation::Launch);
    assert_eq!(errors(&launched), Vec::<String>::new());
    let json = serde_json::to_value(launched.mesh_draw.as_ref().unwrap()).unwrap();
    assert_eq!(json["observation"], serde_json::json!({"kind": "launch"}));
    let mut threads = launched.clone();
    threads.mesh_draw.as_mut().unwrap().grid = MeshGrid::Threads([3, 2, 1]);
    assert_eq!(errors(&threads), Vec::<String>::new());

    let mut partnered = launched.clone();
    partnered.mesh_draw.as_mut().unwrap().mesh = Some("mesh_main".into());
    refused(&partnered, "names a mesh partner and a launch observation");
    for grid in [
        MeshGrid::Threadgroups([1, 2, 1]),
        MeshGrid::Threads([5, 1, 1]),
    ] {
        let mut two = launched.clone();
        two.mesh_draw.as_mut().unwrap().grid = grid;
        refused(&two, "launch observation needs one object threadgroup");
    }
    let mut mesh = mesh_case();
    let draw = mesh.mesh_draw.as_mut().unwrap();
    draw.fragment = None;
    draw.observation = Some(MeshObservation::Launch);
    refused(&mesh, "launch observation of a mesh case");
}

/// A mesh case may state what a generated object function launches in its object partner's place:
/// whole words of payload up to Metal's 16 KB, and at most 1024 mesh threadgroups; never beside
/// an object partner, or in an object case, whose entry is its object function.
#[test]
fn an_object_launch_takes_the_object_partner_s_place() {
    let launch = |bytes: &[u8], mesh_threadgroups| ObjectLaunch {
        payload_b64: base64::engine::general_purpose::STANDARD.encode(bytes),
        mesh_threadgroups,
    };
    let mut launched = mesh_case();
    launched.mesh_draw.as_mut().unwrap().object_launch = Some(launch(&[0; 32], [2, 1, 1]));
    assert_eq!(errors(&launched), Vec::<String>::new());
    let json = serde_json::to_value(launched.mesh_draw.as_ref().unwrap()).unwrap();
    assert_eq!(
        json["object_launch"],
        serde_json::json!({"payload_b64": base64::engine::general_purpose::STANDARD.encode([0u8; 32]), "mesh_threadgroups": [2, 1, 1]})
    );

    for (bytes, grid, what) in [
        (&[0u8; 6][..], [1, 1, 1], "holds 6 bytes"),
        (&[][..], [1, 1, 1], "holds 0 bytes"),
        (
            &[0u8; MESH_PAYLOAD_BYTES + 4][..],
            [1, 1, 1],
            "past Metal's 16384-byte payload",
        ),
        (&[0u8; 4][..], [64, 32, 1], "holds 2048 threadgroups"),
    ] {
        let mut wrong = launched.clone();
        wrong.mesh_draw.as_mut().unwrap().object_launch = Some(launch(bytes, grid));
        refused(&wrong, what);
    }
    let mut partnered = launched.clone();
    partnered.mesh_draw.as_mut().unwrap().object = Some("object_main".into());
    refused(&partnered, "names an object partner and an object launch");
    let mut object = launched;
    object.stage = Stage::Object;
    object.entry = "object_main".into();
    object.mesh_draw.as_mut().unwrap().mesh = Some("mesh_main".into());
    refused(&object, "mesh_draw.object_launch of an object case");
}

/// Without rasterization the pipeline has no fragment stage and no attachment, so the case
/// observes a buffer of its entry's.
#[test]
fn a_mesh_draw_without_rasterization_observes_a_buffer() {
    let mut off = mesh_case();
    off.mesh_draw.as_mut().unwrap().rasterization = false;
    refused(&off, "mesh_draw.fragment without rasterization");
    refused(&off, "does not accept render targets");
    refused(&off, "must select a shader resource");
    off.mesh_draw.as_mut().unwrap().fragment = None;
    off.render_targets.clear();
    off.buffers = example().buffers;
    off.output = example().output;
    assert_eq!(errors(&off), Vec::<String>::new());
}

/// What a mesh case does not author: a mesh function's threadgroup memory (Metal has no setter
/// for it), an indirect command buffer, a fragment case's attachments and pass, and what no object
/// or mesh stage binds (an acceleration structure, Metal having no setter for one there).
#[test]
fn a_mesh_case_authors_no_other_stage_s_work() {
    let mut threadgroup = mesh_case();
    threadgroup.threadgroup_memory = vec![ThreadgroupMemoryResource {
        binding: 0,
        length: 16,
    }];
    refused(
        &threadgroup,
        "Metal sets no mesh function's threadgroup memory",
    );
    let mut depth = mesh_case();
    depth.depth_stencil = Some(DepthStencilResource {
        dimensions: [8, 8],
        initial_depth_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 8 * 8 * 4])),
        initial_stencil_b64: None,
        state: None,
    });
    refused(&depth, "depth_stencil is valid only for fragment cases");
    let mut samples = mesh_case();
    samples.raster_sample_count = Some(4);
    refused(
        &samples,
        "raster_sample_count is valid only for a fragment case's draw",
    );
    let mut acceleration = mesh_case();
    acceleration
        .acceleration_structures
        .push(AccelerationStructureResource {
            binding: 3,
            kind: AccelerationStructureKind::Instance,
            primitive_triangles_b64: None,
            child_references: vec![0x1234],
        });
    refused(
        &acceleration,
        "acceleration_structures is not modelled for an object or mesh case",
    );
}
