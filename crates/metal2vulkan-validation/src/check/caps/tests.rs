use super::*;
use crate::case::{
    DepthStencilResource, Dispatch, Draw, FragmentImageblockResource, Primitive,
    RenderTargetResource, TessellationFactors, TextureFormat, ThreadgroupMemoryResource,
};
use base64::Engine as _;

fn kernel() -> AuthoredCase {
    serde_json::from_value(serde_json::json!({
        "air_sha256": "00".repeat(32),
        "case_id": "00".repeat(32),
        "name": "caps",
        "entry": "main",
        "stage": "kernel",
        "dispatch": {"grid": [1, 1, 1], "threads_per_threadgroup": [1, 1, 1]},
        "output": {"kind": "none"},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free"
    }))
    .unwrap()
}

fn errors(case: &AuthoredCase) -> String {
    cap_errors(case).join("\n")
}

#[test]
fn a_small_case_is_within_every_cap() {
    assert_eq!(cap_errors(&kernel()), Vec::<String>::new());
}

#[test]
fn dispatch_invocations_cap_at_two_to_the_twentieth() {
    let mut case = kernel();
    case.dispatch = Some(Dispatch {
        grid: [1024, 32, 32],
        threads_per_threadgroup: [32, 1, 1],
        stage_in_region: None,
    });
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.dispatch.as_mut().unwrap().grid = [1024 * 32 * 32 + 1, 1, 1];
    assert!(
        errors(&case).contains("1048577 invocations"),
        "{}",
        errors(&case)
    );
}

#[test]
fn threadgroup_size_caps_at_1024() {
    let mut case = kernel();
    case.dispatch = Some(Dispatch {
        grid: [1024, 1, 1],
        threads_per_threadgroup: [16, 16, 4],
        stage_in_region: None,
    });
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.dispatch.as_mut().unwrap().threads_per_threadgroup = [1025, 1, 1];
    assert!(errors(&case).contains("1025 threads"), "{}", errors(&case));
}

#[test]
fn draw_vertices_times_instances_cap_at_two_to_the_sixteenth() {
    let mut case = kernel();
    case.dispatch = None;
    case.draw = Some(Draw {
        primitive: Primitive::Triangle,
        vertex_start: 0,
        vertex_count: 1 << 8,
        instance_count: 1 << 8,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.draw.as_mut().unwrap().vertex_count = (1 << 16) + 1;
    case.draw.as_mut().unwrap().instance_count = 1;
    assert!(errors(&case).contains("65537"), "{}", errors(&case));
}

#[test]
fn tessellated_draws_are_capped_by_their_evaluated_vertex_bound() {
    const ONE: u16 = 0x3c00;
    const FOUR: u16 = 0x4400;
    let patch = |factor| TessellationFactors {
        edge_f16: vec![factor; 3],
        inside_f16: vec![factor],
    };
    let mut case = kernel();
    case.dispatch = None;
    // A factor-1 patch has at most (1 + 1)^2 = 4 domain points.
    case.tessellation = Some(TessellationDraw {
        factors: vec![patch(ONE)],
        instance_count: 1 << 14,
        amplification_count: 1,
        control_points: vec![],
        patch_inputs: vec![],
        partition_mode: Default::default(),
        output_winding: Default::default(),
        cull_mode: Default::default(),
    });
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.tessellation.as_mut().unwrap().instance_count = (1 << 14) + 1;
    assert!(errors(&case).contains("65540"), "{}", errors(&case));
    // A factor-4 patch has at most 25; 2621 instances stay at 65525.
    let tessellation = case.tessellation.as_mut().unwrap();
    tessellation.factors = vec![patch(FOUR)];
    tessellation.instance_count = 2621;
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.tessellation.as_mut().unwrap().instance_count = 2622;
    assert!(errors(&case).contains("65550"), "{}", errors(&case));
    // An infinite factor is clamped to the maximum factor, 64: (64 + 1)^2 = 4225 points, and a
    // factor beyond it the same.
    let tessellation = case.tessellation.as_mut().unwrap();
    tessellation.instance_count = 15;
    for factor in [0x7c00, 0x5640] {
        case.tessellation.as_mut().unwrap().factors = vec![patch(factor)];
        assert_eq!(cap_errors(&case), Vec::<String>::new());
        case.tessellation.as_mut().unwrap().instance_count = 16;
        assert!(errors(&case).contains("67600"), "{}", errors(&case));
        case.tessellation.as_mut().unwrap().instance_count = 15;
    }
    // A patch an edge factor of which culls evaluates nothing, whatever its other factors: zero,
    // negative, and NaN.
    for culling in [0x0000, 0x8000, 0xbc00, 0x7e00] {
        let mut culled = patch(0x7c00);
        culled.edge_f16[1] = culling;
        let tessellation = case.tessellation.as_mut().unwrap();
        tessellation.factors = vec![culled; 64];
        tessellation.instance_count = 1 << 14;
        assert_eq!(cap_errors(&case), Vec::<String>::new(), "{culling:#06x}");
    }
    // A NaN inside factor takes the lower end; pow2 rounds 33 segments up to 64, and fractional
    // even starts at 2.
    let tessellation = case.tessellation.as_mut().unwrap();
    tessellation.factors = vec![TessellationFactors {
        edge_f16: vec![ONE; 3],
        inside_f16: vec![0x7e00],
    }];
    tessellation.instance_count = 1 << 14;
    tessellation.partition_mode = crate::case::TessellationPartition::FractionalEven;
    assert!(errors(&case).contains("147456"), "{}", errors(&case));
    let tessellation = case.tessellation.as_mut().unwrap();
    tessellation.partition_mode = crate::case::TessellationPartition::Pow2;
    tessellation.instance_count = 1;
    tessellation.factors = vec![patch(0x5020)];
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.tessellation.as_mut().unwrap().instance_count = 16;
    assert!(errors(&case).contains("67600"), "{}", errors(&case));
    // A fractional partition splits a factor into as many segments as the next odd (fractional
    // odd) or even (fractional even) integer: 3.0 is four segments under fractional even, so
    // 4096 instances are (4 + 1)^2 x 4096 = 102400 points, not (3 + 1)^2 x 4096 = 65536; 2.0 is
    // three under fractional odd, so 7281 instances are 16 x 7281 = 116496, not 65529.
    use crate::case::TessellationPartition as P;
    for (mode, factor, instances, points) in [
        (P::FractionalEven, 0x4200, 4096, "102400"),
        (P::FractionalOdd, 0x4000, 7281, "116496"),
    ] {
        let tessellation = case.tessellation.as_mut().unwrap();
        tessellation.partition_mode = mode;
        tessellation.factors = vec![patch(factor)];
        tessellation.instance_count = instances;
        assert!(
            errors(&case).contains(points),
            "{mode:?}: {}",
            errors(&case)
        );
    }
    // Fractional odd clamps to max - 1 = 63, already odd: (63 + 1)^2 = 4096 points, 16 instances
    // within the cap and 17 over it; fractional even to 64, (64 + 1)^2 = 4225, 16 over it.
    for (mode, within, over, points) in [
        (P::FractionalOdd, 16, 17, "69632"),
        (P::FractionalEven, 15, 16, "67600"),
    ] {
        let tessellation = case.tessellation.as_mut().unwrap();
        tessellation.partition_mode = mode;
        tessellation.factors = vec![patch(0x7c00)];
        tessellation.instance_count = within;
        assert_eq!(cap_errors(&case), Vec::<String>::new(), "{mode:?}");
        case.tessellation.as_mut().unwrap().instance_count = over;
        assert!(
            errors(&case).contains(points),
            "{mode:?}: {}",
            errors(&case)
        );
    }
}

#[test]
fn attachments_cap_at_1024_by_1024() {
    let mut case = kernel();
    case.render_targets = vec![RenderTargetResource {
        index: 0,
        format: TextureFormat::R32Uint,
        dimensions: [1024, 1024],
        initial_bytes_b64: String::new(),
    }];
    case.depth_stencil = Some(DepthStencilResource {
        dimensions: [1024, 1024],
        initial_depth_b64: None,
        initial_stencil_b64: None,
        state: None,
    });
    case.fragment_imageblock = Some(FragmentImageblockResource {
        dimensions: [1024, 1024],
        members: vec![],
    });
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.render_targets[0].dimensions = [1025, 1];
    case.depth_stencil.as_mut().unwrap().dimensions = [1, 1025];
    case.fragment_imageblock.as_mut().unwrap().dimensions = [1025, 1025];
    let errors = errors(&case);
    assert!(errors.contains("render target 0 extent 1025x1"), "{errors}");
    assert!(
        errors.contains("depth/stencil attachment extent 1x1025"),
        "{errors}"
    );
    assert!(
        errors.contains("fragment imageblock extent 1025x1025"),
        "{errors}"
    );
}

#[test]
fn resource_bytes_cap_at_256_mib_across_literal_and_allocated_storage() {
    let mut case = kernel();
    // 12 literal bytes in one buffer, the rest in threadgroup memory: exactly the cap.
    case.buffers = vec![crate::case::BufferResource {
        binding: 0,
        role: crate::case::ResourceRole::Input,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([7u8; 12])),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    }];
    let remaining = MAX_RESOURCE_BYTES - 12;
    case.threadgroup_memory = vec![
        ThreadgroupMemoryResource {
            binding: 0,
            length: u32::try_from(remaining / 2).unwrap(),
        },
        ThreadgroupMemoryResource {
            binding: 1,
            length: u32::try_from(remaining - remaining / 2).unwrap(),
        },
    ];
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.threadgroup_memory[1].length += 1;
    assert!(
        errors(&case).contains(&format!("{} bytes", MAX_RESOURCE_BYTES + 1)),
        "{}",
        errors(&case)
    );
}

#[test]
fn literal_bytes_count_every_b64_field_at_any_depth() {
    let manifest = serde_json::json!({
        "buffers": [{"bytes_b64": "AAECAw=="}, {"initial_bytes_b64": "AAEC"}],
        "nested": {"elements": [{"bytes_b64": "AA=="}]},
        "name": "AAECAw==",
    });
    assert_eq!(literal_bytes(&manifest), 4 + 3 + 1);
}

#[test]
fn indirect_command_buffers_cap_at_eight() {
    use crate::case::indirect_commands::{CommandDescriptor, ComputeDispatch};
    let command_buffer = crate::case::indirect_commands::CommandBufferResource {
        buffer_binding: 0,
        field_offset: 0,
        capacity: 1,
        execution_range: [0, 1],
        inherit_pipeline_state: false,
        inherit_buffers: false,
        descriptor: CommandDescriptor::Compute {
            dispatch: ComputeDispatch::Both,
            max_buffer_bind_count: 0,
            max_threadgroup_memory_bind_count: 0,
            support_dynamic_attribute_stride: false,
        },
        commands: vec![],
        execute: None,
    };
    let mut case = kernel();
    case.command_buffers = (0..8)
        .map(|index| {
            let mut buffer = command_buffer.clone();
            buffer.field_offset = index * 8;
            buffer
        })
        .collect();
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.command_buffers.push(command_buffer);
    assert!(errors(&case).contains("9 indirect command buffers"));
}

#[test]
fn declared_fuel_caps_at_two_to_the_twenty_fourth() {
    let mut case = kernel();
    case.fuel = Some(MAX_CASE_FUEL);
    assert_eq!(cap_errors(&case), Vec::<String>::new());
    case.fuel = Some(MAX_CASE_FUEL + 1);
    assert!(errors(&case).contains("fuel 16777217"));
    case.fuel = Some(0);
    assert!(errors(&case).contains("fuel 0"));
}

#[test]
fn declared_fuel_is_part_of_the_semantic_identity_only_when_present() {
    let case = kernel();
    let without = case.computed_case_id().unwrap();
    let mut fueled = case.clone();
    fueled.fuel = Some(1 << 20);
    assert_ne!(fueled.computed_case_id().unwrap(), without);
    let json = serde_json::to_value(&case).unwrap();
    assert!(json.get("fuel").is_none(), "absent fuel must not serialize");
}

/// A fragment case's imageblock cells, one reflected cell per pixel of its extent, count toward the
/// resource cap with its literal bytes, since the executors allocate them; the cell is the data
/// size or the last member's end, rounded up to a word.
#[test]
fn fragment_imageblock_cells_count_toward_the_resource_cap() {
    use crate::air::reflect::{FragmentImageblock, FragmentImageblockMember, ResourceAccess};
    let reflected = |sample_size, member_end: u32| FragmentImageblock {
        sample_size,
        members: vec![FragmentImageblockMember {
            offset: member_end - 2,
            size: 2,
            type_name: "half".into(),
            semantic: "user(a)".into(),
            raster_order_group: 0,
            binding: Some(0),
            access: ResourceAccess::ReadWrite,
        }],
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    assert_eq!(fragment_imageblock_cell_bytes(&reflected(8, 2)), 8);
    assert_eq!(fragment_imageblock_cell_bytes(&reflected(2, 6)), 8);
    assert_eq!(fragment_imageblock_cell_bytes(&reflected(9, 2)), 12);
    let mut case = kernel();
    assert_eq!(
        fragment_imageblock_cap_errors(&case, &reflected(1 << 12, 2)),
        Vec::<String>::new()
    );
    case.fragment_imageblock = Some(FragmentImageblockResource {
        dimensions: [1024, 1024],
        members: vec![],
    });
    // 256-byte cells over 1024x1024 pixels are exactly the cap.
    assert_eq!(
        fragment_imageblock_cap_errors(&case, &reflected(256, 2)),
        Vec::<String>::new()
    );
    let errors = fragment_imageblock_cap_errors(&case, &reflected(257, 2)).join("\n");
    assert!(
        errors.contains(&format!("{} bytes", 1024 * 1024 * 260)) && errors.contains("260-byte"),
        "{errors}"
    );
    // A data size from the AIR is not bounded by the manifest: a huge one is refused.
    assert!(!fragment_imageblock_cap_errors(&case, &reflected(u32::MAX, 2)).is_empty());
}
