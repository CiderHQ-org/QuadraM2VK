use super::*;
use crate::case::out_of_bounds::{
    ContainmentForm, OutOfBounds, OutOfBoundsRead, OutOfBoundsResource, OutOfBoundsTarget,
    MAX_REACH,
};

/// [`example`] with a 16-byte input buffer 1 it reads out of bounds into the 32-byte output.
fn out_of_bounds_case() -> AuthoredCase {
    let mut case = example();
    case.buffers[0].initial_bytes_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode([0xab; 32]));
    case.output = OutputSelection::Buffer {
        binding: 0,
        offset: 0,
        length: 32,
    };
    case.buffers.push(BufferResource {
        binding: 1,
        role: ResourceRole::Input,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([1; 16])),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    });
    case.out_of_bounds = Some(OutOfBounds {
        targets: vec![OutOfBoundsTarget {
            resource: OutOfBoundsResource::Buffer { binding: 1 },
            form: ContainmentForm::StorageBuffer,
            reach_before: 0,
            reach_after: 256,
        }],
        reads: vec![OutOfBoundsRead {
            offset: 16,
            length: 16,
            target: 0,
        }],
    });
    case.case_id = case.computed_case_id().unwrap();
    case
}

fn errors(case: &AuthoredCase) -> String {
    case.validate_literal_resources()
        .err()
        .unwrap_or_default()
        .join("\n")
}

#[test]
fn an_out_of_bounds_declaration_is_part_of_the_identity() {
    let case = out_of_bounds_case();
    assert_eq!(errors(&case), "");
    let mut without = case.clone();
    without.out_of_bounds = None;
    assert_ne!(without.computed_case_id().unwrap(), case.case_id);
    let mut farther = case.clone();
    farther.out_of_bounds.as_mut().unwrap().targets[0].reach_after = 512;
    assert_ne!(farther.computed_case_id().unwrap(), case.case_id);
    // A case without a declaration serializes as before the field existed.
    assert!(!serde_json::to_string(&without)
        .unwrap()
        .contains("out_of_bounds"));
}

#[test]
fn a_target_is_a_declared_resource_of_its_form() {
    let edit = |change: &dyn Fn(&mut OutOfBounds)| {
        let mut case = out_of_bounds_case();
        change(case.out_of_bounds.as_mut().unwrap());
        case.case_id = case.computed_case_id().unwrap();
        errors(&case)
    };
    assert!(edit(&|d| d.targets.clear()).contains("declares no target"));
    assert!(
        edit(&|d| d.targets[0].resource = OutOfBoundsResource::Buffer { binding: 5 })
            .contains("declares no such resource")
    );
    assert!(edit(&|d| d.targets[0].form = ContainmentForm::Texture)
        .contains("not a form of this resource"));
    assert!(edit(&|d| d.targets[0].form = ContainmentForm::VertexInput)
        .contains("form vertex_input is authorable in Vertex cases only"));
    assert!(edit(&|d| d.targets.push(d.targets[0].clone())).contains("declared twice"));
    assert!(edit(&|d| d.targets[0].reach_after = 0).contains("reach_after must be a nonzero"));
    assert!(edit(&|d| d.targets[0].reach_after = 24).contains("reach_after must be a nonzero"));
    assert!(edit(&|d| d.targets[0].reach_before = 16).contains("reach_before must be a multiple"));
    assert!(edit(&|d| d.targets[0].reach_after = 2 * MAX_REACH)
        .contains(&format!("exceeds {MAX_REACH}")));
    assert_eq!(edit(&|d| d.targets[0].reach_before = 256), "");
}

#[test]
fn a_buffer_target_is_its_own_padded_allocation() {
    let mut case = out_of_bounds_case();
    case.buffers[1].bytes_b64 = Some(base64::engine::general_purpose::STANDARD.encode([1; 20]));
    case.case_id = case.computed_case_id().unwrap();
    assert!(errors(&case).contains("multiple of 16 bytes"));
}

/// A rasterization-rate map's buffer is its executor's own map data, not bytes the case
/// authored, so it is no buffer target (the RRMb verifier's seventh pass, third low).
#[test]
fn a_rate_map_buffer_is_no_buffer_target() {
    assert!(edited(&out_of_bounds_case(), &|case| {
        let buffer = &mut case.buffers[1];
        buffer.bytes_b64 = None;
        buffer.rasterization_rate_map = Some(crate::case::RateMapLiteral {
            screen_size: [37, 23],
            layers: vec![crate::case::RateMapLayerLiteral {
                horizontal: vec![crate::case::RateSample(1.0)],
                vertical: vec![crate::case::RateSample(1.0)],
            }],
        });
    })
    .contains("a rasterization-rate map's buffer is no out-of-bounds target"));
}

#[test]
fn a_read_lies_in_the_output_and_reads_a_shadowed_target() {
    let edit = |change: &dyn Fn(&mut OutOfBounds)| {
        let mut case = out_of_bounds_case();
        change(case.out_of_bounds.as_mut().unwrap());
        case.case_id = case.computed_case_id().unwrap();
        errors(&case)
    };
    assert!(edit(&|d| d.reads[0].target = 1).contains("names no target 1"));
    assert!(edit(&|d| d.reads[0].length = 32).contains("past the selected output"));
    assert!(edit(&|d| d.reads[0].offset = 2).contains("multiples of 4"));
    assert!(edit(&|d| d.reads.push(OutOfBoundsRead {
        offset: 28,
        length: 4,
        target: 0
    }))
    .contains("overlaps another read"));
}

#[test]
fn a_texture_target_compares_with_metal_and_has_no_shadow() {
    let mut case = out_of_bounds_case();
    case.textures.push(TextureResource {
        binding: 0,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::R32Uint,
        dimensions: [2, 2, 1],
        sample_count: 1,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0; 16])),
        initial_bytes_b64: None,
        sparse: None,
        mip_levels: 1,
    });
    let declaration = case.out_of_bounds.as_mut().unwrap();
    declaration.targets[0] = OutOfBoundsTarget {
        resource: OutOfBoundsResource::Texture { binding: 0 },
        form: ContainmentForm::Texture,
        reach_before: 0,
        reach_after: 0,
    };
    case.case_id = case.computed_case_id().unwrap();
    let errors_of = errors(&case);
    assert!(
        errors_of.contains("compares it with Metal, not as a read"),
        "{errors_of}"
    );
    case.out_of_bounds.as_mut().unwrap().reads.clear();
    case.case_id = case.computed_case_id().unwrap();
    assert_eq!(errors(&case), "");
    case.out_of_bounds.as_mut().unwrap().targets[0].form = ContainmentForm::TexelBuffer;
    case.case_id = case.computed_case_id().unwrap();
    assert!(errors(&case).contains("not a form of this resource"));
    case.out_of_bounds.as_mut().unwrap().targets[0].form = ContainmentForm::Texture;
    case.out_of_bounds.as_mut().unwrap().targets[0].reach_after = 16;
    case.case_id = case.computed_case_id().unwrap();
    assert!(errors(&case).contains("an image's memory is opaque"));
}

/// A texture buffer is host-visible buffer memory in the candidate, which guards it: its target
/// declares the guard's reach past it, covering the furthest texel the case accesses, and none
/// before it.
#[test]
fn a_texel_buffer_target_declares_its_guards_reach_past_it() {
    let mut case = out_of_bounds_case();
    case.textures.push(TextureResource {
        binding: 0,
        role: ResourceRole::Input,
        texture_type: TextureType::Buffer,
        format: TextureFormat::R32Uint,
        dimensions: [4, 1, 1],
        sample_count: 1,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0; 16])),
        initial_bytes_b64: None,
        sparse: None,
        mip_levels: 1,
    });
    let declaration = case.out_of_bounds.as_mut().unwrap();
    declaration.reads.clear();
    declaration.targets[0] = OutOfBoundsTarget {
        resource: OutOfBoundsResource::Texture { binding: 0 },
        form: ContainmentForm::TexelBuffer,
        reach_before: 0,
        reach_after: 0,
    };
    let with = |case: &AuthoredCase, before, after| {
        let mut case = case.clone();
        let target = &mut case.out_of_bounds.as_mut().unwrap().targets[0];
        target.reach_before = before;
        target.reach_after = after;
        case.case_id = case.computed_case_id().unwrap();
        errors(&case)
    };
    assert!(with(&case, 0, 0).contains("covering the furthest texel"));
    assert!(with(&case, 0, 24).contains("covering the furthest texel"));
    assert!(with(&case, 256, 16384).contains("nothing lies before it"));
    assert!(with(&case, 0, 2 * MAX_REACH).contains("exceeds"));
    assert_eq!(with(&case, 0, 16384), "");
}

#[test]
fn a_threadgroup_target_is_the_one_allocation() {
    let mut case = out_of_bounds_case();
    case.threadgroup_memory.push(ThreadgroupMemoryResource {
        binding: 0,
        length: 64,
    });
    case.out_of_bounds.as_mut().unwrap().targets[0] = OutOfBoundsTarget {
        resource: OutOfBoundsResource::ThreadgroupMemory { binding: 0 },
        form: ContainmentForm::Threadgroup,
        reach_before: 0,
        reach_after: 64,
    };
    case.case_id = case.computed_case_id().unwrap();
    assert_eq!(errors(&case), "");
    case.threadgroup_memory.push(ThreadgroupMemoryResource {
        binding: 1,
        length: 16,
    });
    case.case_id = case.computed_case_id().unwrap();
    assert!(errors(&case).contains("one threadgroup allocation"));
}

/// Every form has a stable name and is authorable in the cases of its stage: a draw's forms in
/// vertex cases, every other form in kernel cases.
#[test]
fn every_form_has_a_stable_name_and_is_authorable_in_its_stage() {
    for form in ContainmentForm::ALL {
        assert_eq!(
            serde_json::to_string(&form).unwrap(),
            format!("\"{}\"", form.as_str())
        );
        let draw = matches!(
            form,
            ContainmentForm::VertexInput
                | ContainmentForm::ControlPoint
                | ContainmentForm::IndexBuffer
        );
        assert_eq!(
            form.stage(),
            if draw { Stage::Vertex } else { Stage::Kernel },
            "{form}"
        );
    }
    let mut case = out_of_bounds_case();
    case.stage = Stage::Vertex;
    case.case_id = case.computed_case_id().unwrap();
    assert!(errors(&case).contains("form storage_buffer is authorable in Kernel cases only"));
}

#[test]
fn an_out_of_bounds_case_encodes_no_indirect_commands() {
    let mut case = out_of_bounds_case();
    case.pipeline_states
        .push(crate::case::indirect_commands::PipelineResource {
            buffer_binding: 0,
            field_offset: 0,
            token: 1,
            air_sha256: "00".repeat(32),
            case_id: "11".repeat(32),
        });
    case.case_id = case.computed_case_id().unwrap();
    assert!(errors(&case).contains("encodes no indirect commands"));
}

/// [`out_of_bounds_case`] whose buffer 1 is a 32-byte argument buffer holding two
/// argument-buffer textures at bytes 0 and 8 and, at byte 16, the address of a 16-byte
/// argument-buffer buffer (the literal checker reads no AIR, so one case can hold both kinds),
/// with its one target replaced by `target`.
fn embedded_case(target: OutOfBoundsTarget) -> AuthoredCase {
    let mut case = out_of_bounds_case();
    case.buffers[1].bytes_b64 = Some(base64::engine::general_purpose::STANDARD.encode([1; 32]));
    case.argument_buffer_buffers
        .push(ArgumentBufferBufferResource {
            buffer_binding: 1,
            field_offset: 16,
            role: ResourceRole::Input,
            bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([2; 16])),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        });
    for field_offset in [0, 8] {
        case.argument_buffer_textures
            .push(ArgumentBufferTextureResource {
                buffer_binding: 1,
                field_offset,
                role: ResourceRole::Input,
                texture_type: TextureType::D2,
                format: TextureFormat::R32Uint,
                dimensions: [1, 1, 1],
                sample_count: 1,
                bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([3; 4])),
                initial_bytes_b64: None,
            });
    }
    case.out_of_bounds.as_mut().unwrap().targets[0] = target;
    case.case_id = case.computed_case_id().unwrap();
    case
}

fn edited(case: &AuthoredCase, change: &dyn Fn(&mut AuthoredCase)) -> String {
    let mut case = case.clone();
    change(&mut case);
    case.case_id = case.computed_case_id().unwrap();
    errors(&case)
}

/// A buffer an argument buffer points to is shadowed and guarded as a buffer target is: one of the
/// case's argument-buffer buffers, a multiple of 16 bytes, with a reach past it.
#[test]
fn a_loaded_pointer_target_is_a_shadowed_argument_buffer_buffer() {
    let case = embedded_case(OutOfBoundsTarget {
        resource: OutOfBoundsResource::ArgumentBufferBuffer {
            buffer_binding: 1,
            field_offset: 16,
        },
        form: ContainmentForm::LoadedPointer,
        reach_before: 0,
        reach_after: 256,
    });
    assert_eq!(errors(&case), "");
    let target = |change: &dyn Fn(&mut OutOfBoundsTarget)| {
        edited(&case, &|case| {
            change(&mut case.out_of_bounds.as_mut().unwrap().targets[0])
        })
    };
    assert!(target(&|t| t.form = ContainmentForm::StorageBuffer)
        .contains("not a form of this resource"));
    assert!(target(
        &|t| t.resource = OutOfBoundsResource::ArgumentBufferBuffer {
            buffer_binding: 1,
            field_offset: 24
        }
    )
    .contains("declares no such argument-buffer buffer"));
    assert!(target(&|t| t.reach_after = 0).contains("reach_after must be a nonzero"));
    assert!(target(&|t| t.reach_before = 16).contains("reach_before must be a multiple"));
    assert_eq!(target(&|t| t.reach_before = 256), "");
    assert!(edited(&case, &|case| {
        case.argument_buffer_buffers[0].bytes_b64 =
            Some(base64::engine::general_purpose::STANDARD.encode([2; 20]))
    })
    .contains("multiple of 16 bytes"));
    // A rasterization-rate map's buffer is its executor's own map data.
    assert!(edited(&case, &|case| {
        let member = &mut case.argument_buffer_buffers[0];
        member.bytes_b64 = None;
        member.rasterization_rate_map = Some(crate::case::RateMapLiteral {
            screen_size: [37, 23],
            layers: vec![crate::case::RateMapLayerLiteral {
                horizontal: vec![crate::case::RateSample(1.0)],
                vertical: vec![crate::case::RateSample(1.0)],
            }],
        });
    })
    .contains("a rasterization-rate map's buffer is no out-of-bounds target"));
}

/// A thread-local array and an argument buffer's handle array lie in the case's own layout: neither
/// declares a reach, and a handle array's first element is one of the case's argument-buffer
/// textures.
#[test]
fn an_in_layout_target_declares_no_reach() {
    let thread = embedded_case(OutOfBoundsTarget {
        resource: OutOfBoundsResource::ThreadMemory,
        form: ContainmentForm::SizedArray,
        reach_before: 0,
        reach_after: 0,
    });
    let handles = embedded_case(OutOfBoundsTarget {
        resource: OutOfBoundsResource::ArgumentBufferTextureArray {
            buffer_binding: 1,
            field_offset: 0,
        },
        form: ContainmentForm::HandleArrayIndex,
        reach_before: 0,
        reach_after: 0,
    });
    for case in [&thread, &handles] {
        assert_eq!(errors(case), "");
        let target = |change: &dyn Fn(&mut OutOfBoundsTarget)| {
            edited(case, &|case| {
                change(&mut case.out_of_bounds.as_mut().unwrap().targets[0])
            })
        };
        assert!(target(&|t| t.reach_after = 16).contains("must be 0"));
        assert!(target(&|t| t.reach_before = 256).contains("must be 0"));
        assert!(target(&|t| t.form = ContainmentForm::LoadedPointer)
            .contains("not a form of this resource"));
    }
    assert!(edited(&handles, &|case| {
        case.out_of_bounds.as_mut().unwrap().targets[0].resource =
            OutOfBoundsResource::ArgumentBufferTextureArray {
                buffer_binding: 1,
                field_offset: 16,
            }
    })
    .contains("no argument-buffer texture at the array's first element"));
    // The resource names in the declaration.
    let json = serde_json::to_string(&thread.out_of_bounds).unwrap();
    assert!(json.contains(r#"{"kind":"thread_memory"}"#), "{json}");
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// A vertex case drawing 8 points whose one `uint` vertex input at location 0 holds 4 elements
/// (16 bytes), its one target the input's buffer with 16 bytes of reach past it, and the 16
/// output bytes the vertices past it write declared reads.
fn vertex_input_case() -> AuthoredCase {
    let mut case = crate::case::narrow_vertex_test_case("11".repeat(32), "vertex_main".into());
    case.buffers[0].initial_bytes_b64 = Some(b64(&[0; 32]));
    case.vertex_inputs = vec![AttributeInput {
        location: 0,
        format: AttributeFormat::Uint,
        stride: 4,
        dynamic_stride: false,
        bytes_b64: b64(&[1; 16]),
    }];
    case.draw.as_mut().unwrap().vertex_count = 8;
    case.output = OutputSelection::Buffer {
        binding: 0,
        offset: 0,
        length: 32,
    };
    case.out_of_bounds = Some(OutOfBounds {
        targets: vec![OutOfBoundsTarget {
            resource: OutOfBoundsResource::AttributeBuffer { location: 0 },
            form: ContainmentForm::VertexInput,
            reach_before: 0,
            reach_after: 16,
        }],
        reads: vec![OutOfBoundsRead {
            offset: 16,
            length: 16,
            target: 0,
        }],
    });
    case.case_id = case.computed_case_id().unwrap();
    case
}

/// An attribute buffer target is its own padded allocation, fetched from its start, whose reach
/// past it covers exactly the elements the draw fetches past it; without the target, the input must
/// hold every element the draw fetches.
#[test]
fn a_vertex_input_target_reaches_past_the_vertices_it_fetches() {
    let case = vertex_input_case();
    assert_eq!(errors(&case), "");
    let target = |change: &dyn Fn(&mut OutOfBoundsTarget)| {
        edited(&case, &|case| {
            change(&mut case.out_of_bounds.as_mut().unwrap().targets[0])
        })
    };
    assert!(target(&|t| t.reach_after = 0).contains("reach_after must be a nonzero"));
    assert!(target(&|t| t.reach_before = 256).contains("reach_before must be 0"));
    assert!(target(&|t| t.form = ContainmentForm::StageInput)
        .contains("authorable in Kernel cases only"));
    assert!(
        target(&|t| t.resource = OutOfBoundsResource::AttributeBuffer { location: 3 })
            .contains("declares no such resource")
    );
    assert!(
        edited(&case, &|case| case.draw.as_mut().unwrap().vertex_count = 12)
            .contains("beyond its reach_after 16")
    );
    assert!(
        edited(&case, &|case| case.draw.as_mut().unwrap().vertex_count = 4)
            .contains("never past it")
    );
    assert!(edited(&case, &|case| case.vertex_inputs[0].bytes_b64 =
        b64(&[1; 12]))
    .contains("multiple of 16 bytes"));
    assert!(edited(&case, &|case| {
        case.vertex_inputs[0].stride = 12;
        case.vertex_inputs[0].bytes_b64 = b64(&[1; 32]);
    })
    .contains("multiple of its stride 12"));
    assert!(
        edited(&case, &|case| case.out_of_bounds = None).contains("execution requires at least 32")
    );
}

/// [`vertex_input_case`] as an indexed draw of 16 `uint32` indices from a buffer holding 4 (0 to
/// 3), with no vertex input: its one target the index buffer, 48 bytes of reach past it.
fn index_buffer_case() -> AuthoredCase {
    let mut case = vertex_input_case();
    case.vertex_inputs.clear();
    let indices: Vec<u8> = (0u32..4).flat_map(u32::to_le_bytes).collect();
    let draw = case.draw.as_mut().unwrap();
    draw.vertex_count = 16;
    draw.index_buffer = Some(IndexBufferLiteral {
        index_type: IndexType::Uint32,
        offset: 0,
        bytes_b64: b64(&indices),
    });
    let declaration = case.out_of_bounds.as_mut().unwrap();
    declaration.targets[0] = OutOfBoundsTarget {
        resource: OutOfBoundsResource::IndexBuffer,
        form: ContainmentForm::IndexBuffer,
        reach_before: 0,
        reach_after: 48,
    };
    case.case_id = case.computed_case_id().unwrap();
    case
}

/// An indexed draw past its index buffer declares the buffer a target (Metal leaves the draw
/// undefined), fetches no vertex attribute (an index past the buffer is an undefined vertex id),
/// and reads index 0 inside the buffer, the contained result of an index read past it.
#[test]
fn an_index_buffer_target_is_an_attribute_free_indexed_draw() {
    let case = index_buffer_case();
    assert_eq!(errors(&case), "");
    assert_eq!(case.vertex_elements(), Some(4));
    let draw =
        |change: &dyn Fn(&mut Draw)| edited(&case, &|case| change(case.draw.as_mut().unwrap()));
    assert!(draw(&|d| d.vertex_count = 20).contains("beyond its reach_after 48"));
    assert!(draw(&|d| d.vertex_start = 1).contains("draw.vertex_start must be 0"));
    assert!(draw(&|d| d.index_buffer.as_mut().unwrap().offset = 2)
        .contains("multiple of the index size"));
    assert!(draw(&|d| {
        let indices: Vec<u8> = (1u32..5).flat_map(u32::to_le_bytes).collect();
        d.index_buffer.as_mut().unwrap().bytes_b64 = b64(&indices);
    })
    .contains("must include index 0"));
    assert!(edited(&case, &|case| case.vertex_inputs =
        vertex_input_case().vertex_inputs)
    .contains("fetches no vertex attribute"));
    assert!(edited(&case, &|case| case.out_of_bounds = None)
        .contains("declare the index buffer an out-of-bounds target"));
    // In range, an indexed draw needs no target; its 16-bit indices from a nonzero offset.
    let in_range = edited(&case, &|case| {
        case.out_of_bounds = None;
        let draw = case.draw.as_mut().unwrap();
        draw.vertex_count = 2;
        draw.index_buffer = Some(IndexBufferLiteral {
            index_type: IndexType::Uint16,
            offset: 2,
            bytes_b64: b64(&[0, 0, 1, 0, 2, 0]),
        });
    });
    assert_eq!(in_range, "");
    // A kernel case draws nothing.
    assert!(edited(&case, &|case| case.stage = Stage::Kernel)
        .contains("an indexed draw is authorable for vertex cases only"));
    // A draw without an index buffer serializes as before the field existed.
    assert!(!serde_json::to_string(&vertex_input_case())
        .unwrap()
        .contains("index_buffer"));
}

/// A kernel stage input's buffer is a target as a vertex input's is, its elements one per thread on
/// x of the grid.
#[test]
fn a_stage_input_target_reaches_past_the_grid_it_fetches() {
    let mut case = out_of_bounds_case();
    case.buffers.truncate(1);
    case.kernel_stage_inputs = vec![AttributeInput {
        location: 0,
        format: AttributeFormat::Uint,
        stride: 4,
        dynamic_stride: false,
        bytes_b64: b64(&[1; 16]),
    }];
    case.dispatch.as_mut().unwrap().grid = [8, 1, 1];
    case.out_of_bounds.as_mut().unwrap().targets[0] = OutOfBoundsTarget {
        resource: OutOfBoundsResource::AttributeBuffer { location: 0 },
        form: ContainmentForm::StageInput,
        reach_before: 0,
        reach_after: 16,
    };
    case.case_id = case.computed_case_id().unwrap();
    assert_eq!(errors(&case), "");
    assert!(edited(&case, &|case| case.dispatch.as_mut().unwrap().grid =
        [16, 1, 1])
    .contains("beyond its reach_after 16"));
    assert!(edited(&case, &|case| {
        case.out_of_bounds.as_mut().unwrap().targets[0].form = ContainmentForm::VertexInput
    })
    .contains("authorable in Vertex cases only"));
}

/// A top-level texture array target lies in the case's own layout: it holds at least two
/// elements, its `reach_after` counts the argument-table slots past its last element that the
/// case's indices reach, and every one of them is bound, so no index of the case names a slot
/// Metal holds nothing in.
#[test]
fn a_texture_array_target_binds_every_slot_its_indices_reach() {
    let mut case = out_of_bounds_case();
    let array = |binding| TextureArrayResource {
        binding,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::R32Uint,
        sample_count: 1,
        overrides_texture_at_base: false,
        elements: vec![
            TextureArrayElement {
                dimensions: [1, 1, 1],
                bytes_b64: Some(b64(&[3; 4])),
                initial_bytes_b64: None,
            };
            2
        ],
    };
    case.texture_arrays = vec![array(0), array(2)];
    let declaration = case.out_of_bounds.as_mut().unwrap();
    declaration.targets[0] = OutOfBoundsTarget {
        resource: OutOfBoundsResource::TextureArray { binding: 0 },
        form: ContainmentForm::HandleArrayIndex,
        reach_before: 0,
        reach_after: 2,
    };
    case.case_id = case.computed_case_id().unwrap();
    assert_eq!(errors(&case), "");
    let reach = |reach_after: u32| {
        move |case: &mut AuthoredCase| {
            case.out_of_bounds.as_mut().unwrap().targets[0].reach_after = reach_after
        }
    };
    // One slot reached: bound by the second array.
    assert_eq!(edited(&case, &reach(1)), "");
    // The second array binds slots 2 and 3; a third slot past the first array is unbound.
    assert!(edited(&case, &reach(3))
        .contains("binds no texture at 1 of the 3 slots past the array's last element"));
    assert!(edited(&case, &|case| case.texture_arrays.truncate(1))
        .contains("binds no texture at 2 of the 2 slots"));
    // No slot reached is no target past the array.
    assert!(edited(&case, &reach(0)).contains("at least 1"));
    assert!(edited(&case, &|case| {
        case.out_of_bounds.as_mut().unwrap().targets[0].reach_before = 1
    })
    .contains("reach_before must be 0"));
    assert!(edited(&case, &|case| {
        case.out_of_bounds.as_mut().unwrap().targets[0].form = ContainmentForm::Texture
    })
    .contains("not a form of this resource"));
}
