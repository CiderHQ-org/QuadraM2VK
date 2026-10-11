use super::*;

#[test]
fn documentation_and_manifest_field_order_do_not_change_identity() {
    let mut case = example();
    let id = case.computed_case_id().unwrap();
    case.rationale = Some("different explanation".into());
    case.authored_by = Some("someone else".into());
    case.name = "renamed-slot".into();
    assert_eq!(case.computed_case_id().unwrap(), id);

    case.case_id = id;
    let json = serde_json::to_value(&case).unwrap();
    let mut entries = json.as_object().unwrap().iter().collect::<Vec<_>>();
    entries.reverse();
    let reversed = serde_json::Value::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    );
    let reparsed: AuthoredCase = serde_json::from_value(reversed).unwrap();
    assert_eq!(reparsed.computed_case_id().unwrap(), case.case_id);
}

#[test]
fn authored_bounded_execution_requires_an_explicit_bound_rationale() {
    let mut case = example();
    case.execution_safety = ExecutionSafety::AuthoredBounded;
    case.rationale = None;
    case.case_id = case.computed_case_id().unwrap();
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("finite loop bound"), "{errors}");

    case.rationale = Some("input count=4 bounds the only loop to four iterations".into());
    case.validate_literal_resources().unwrap();
}

#[test]
fn resource_array_order_does_not_change_semantic_or_input_identity() {
    let mut first = example();
    first.buffers.push(BufferResource {
        binding: 7,
        role: ResourceRole::Input,
        bytes_b64: Some("AQIDBA==".into()),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    });
    first.case_id = first.computed_case_id().unwrap();
    let first_input = first.computed_input_sha256().unwrap();

    let mut reordered = first.clone();
    reordered.buffers.reverse();
    assert_eq!(reordered.computed_case_id().unwrap(), first.case_id);
    assert_eq!(reordered.computed_input_sha256().unwrap(), first_input);
}

#[test]
fn canonical_identity_and_input_digest_are_golden() {
    let mut case = example();
    case.case_id = case.computed_case_id().unwrap();
    assert_eq!(
        case.case_id,
        "209b8f187ed6b052352d63ba7c8445362bab6b49c61e9b96006522bb7d880a82"
    );
    assert_eq!(
        case.computed_input_sha256().unwrap(),
        "f4627a35e4ef25183ec50b1b61fa81231185fbe6792ffa36a41ac9ac9af477ee"
    );
}

#[test]
fn malformed_resources_are_rejected_without_repair() {
    let mut case = example();
    case.buffers[0].initial_bytes_b64 = None;
    case.buffers[0].bytes_b64 = Some("not base64".into());
    case.dispatch.as_mut().unwrap().grid[0] = 0;
    let errors = case.validate_literal_resources().unwrap_err().join("\n");
    assert!(errors.contains("dimensions must be nonzero"), "{errors}");
    assert!(
        errors.contains("requires only initial_bytes_b64"),
        "{errors}"
    );
}

#[test]
fn missing_and_unknown_manifest_fields_are_rejected() {
    let case = example();
    let mut value = serde_json::to_value(case).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("guessed_seed".into(), serde_json::json!(7));
    assert!(serde_json::from_value::<AuthoredCase>(value).is_err());

    let mut value = serde_json::to_value(example()).unwrap();
    value.as_object_mut().unwrap().remove("output");
    assert!(serde_json::from_value::<AuthoredCase>(value).is_err());
}
