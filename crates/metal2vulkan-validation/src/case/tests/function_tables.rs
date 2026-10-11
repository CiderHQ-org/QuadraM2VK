use super::*;

#[test]
fn direct_visible_function_reference_is_semantic_input() {
    let mut case = example();
    let original_case = case.computed_case_id().unwrap();
    case.case_id = original_case.clone();
    let original_input = case.computed_input_sha256().unwrap();
    case.visible_function_references
        .push(LinkedFunctionResource {
            module_sha256: "22".repeat(32),
            function: "linked".into(),
        });
    assert_ne!(case.computed_case_id().unwrap(), original_case);
    assert_ne!(case.computed_input_sha256().unwrap(), original_input);
}

#[test]
fn function_table_capacity_is_independent_of_populated_slots() {
    let mut case = example();
    case.intersection_function_tables
        .push(IntersectionFunctionTableResource {
            binding: 6,
            size: 4,
            entries: vec![],
        });
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();

    case.intersection_function_tables[0].size = 0;
    case.case_id = case.computed_case_id().unwrap();
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("size must be nonzero"), "{errors}");

    case.intersection_function_tables[0] = IntersectionFunctionTableResource {
        binding: 6,
        size: 2,
        entries: vec![IntersectionFunctionTableEntry::Linked {
            index: 2,
            module_sha256: "22".repeat(32),
            function: "intersection".into(),
        }],
    };
    case.case_id = case.computed_case_id().unwrap();
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("entry 2 exceeds size 2"), "{errors}");
}

#[test]
fn opaque_triangle_is_a_typed_populated_intersection_slot() {
    let mut case = example();
    case.intersection_function_tables
        .push(IntersectionFunctionTableResource {
            binding: 6,
            size: 1,
            entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
                index: 0,
                signature: vec![
                    IntersectionFunctionSignature::TriangleData,
                    IntersectionFunctionSignature::IntersectionFunctionBuffer,
                ],
            }],
        });
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    let json = serde_json::to_string(&case).unwrap();
    assert!(json.contains(r#""kind":"opaque_triangle""#));
    assert!(!json.contains("module_sha256"));

    case.intersection_function_tables[0].entries =
        vec![IntersectionFunctionTableEntry::OpaqueTriangle {
            index: 0,
            signature: vec![
                IntersectionFunctionSignature::IntersectionFunctionBuffer,
                IntersectionFunctionSignature::TriangleData,
            ],
        }];
    case.case_id = case.computed_case_id().unwrap();
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(
        errors.contains("signature flags must be sorted and unique"),
        "{errors}"
    );
}

#[test]
fn argument_buffer_intersection_table_uses_owner_and_field_identity() {
    let mut case = example();
    case.argument_buffer_intersection_function_tables.push(
        ArgumentBufferIntersectionFunctionTableResource {
            buffer_binding: 0,
            field_offset: 8,
            size: 1,
            entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
                index: 0,
                signature: vec![IntersectionFunctionSignature::IntersectionFunctionBuffer],
            }],
        },
    );
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();

    case.argument_buffer_intersection_function_tables[0].buffer_binding = 31;
    case.case_id = case.computed_case_id().unwrap();
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("owns an undeclared buffer"), "{errors}");
}

#[test]
fn acceleration_structures_are_semantic_and_cannot_alias_buffers() {
    let mut case = example();
    let original_id = case.computed_case_id().unwrap();
    case.acceleration_structures
        .push(AccelerationStructureResource {
            binding: 0,
            kind: AccelerationStructureKind::Instance,
            primitive_triangles_b64: None,
            child_references: vec![0x1234],
        });
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("both a buffer and an acceleration structure"));
    assert_ne!(original_id, case.computed_case_id().unwrap());
    assert_ne!(
        case.computed_input_sha256().unwrap(),
        example().computed_input_sha256().unwrap()
    );
}

#[test]
fn primitive_acceleration_structures_require_exact_finite_triangle_bytes() {
    let mut case = example();
    case.case_id = "22".repeat(32);
    case.acceleration_structures
        .push(AccelerationStructureResource {
            binding: 5,
            kind: AccelerationStructureKind::Primitive,
            primitive_triangles_b64: Some(
                base64::engine::general_purpose::STANDARD.encode([0u8; 35]),
            ),
            child_references: vec![],
        });
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("36-byte float3 triangles"), "{errors}");

    let mut bytes = [0u8; 36];
    bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    case.acceleration_structures[0].primitive_triangles_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode(bytes));
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("is not finite"), "{errors}");

    case.acceleration_structures[0].primitive_triangles_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode([0u8; 36]));
    if let Err(errors) = case.validate_literal_resources() {
        panic!("{}", errors.join("\n"));
    }
}
