use super::*;

#[test]
fn function_constant_wrapped_function_tables_keep_their_linkage_roles() {
    let ll = r#"
!air.vertex = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.function_constant", !5, !"air.visible_function_table", !"air.location_index", i32 7, i32 1, !"air.read"}
!4 = !{i32 1, !"air.function_constant", !6, !"air.intersection_function_table", !"air.location_index", i32 8, i32 1, !"air.read"}
!5 = !{ptr addrspace(2) @visible_enabled, !"bool", !"visible_enabled"}
!6 = !{ptr addrspace(2) @intersection_enabled, !"bool", !"intersection_enabled"}
"#;
    let meta = parse_air_vertex_meta(ll).0.unwrap();
    assert_eq!(meta.role_of(0), Some(&VertRole::VisibleFunctionTable(7)));
    assert_eq!(
        meta.role_of(1),
        Some(&VertRole::IntersectionFunctionTable(8))
    );
}

/// The fragment decode reads a gated argument's role with `primary_role`, which looks past the
/// `air.function_constant` wrapper unconditionally; the kernel and vertex decodes read theirs with
/// `resource_input_role`. For a resource role the two have to answer the same thing, or one
/// declaration produces a descriptor in one stage and nothing in another -- which is how a gated
/// `air.sampler` came to bind in fragments and vanish in kernels and vertices while the gated
/// `air.texture` beside it bound in all three.
///
/// Checked over the promotion list itself rather than over a copy of it, so a role added to
/// `FC_PROMOTED_RESOURCE_ROLES` is covered by construction. That makes it silent about a role
/// MISSING from the list, which is the defect that happened; the non-vacuous pin for that was the
/// old engine's `tests/gated_system_values.rs::a_gated_sampler_is_a_descriptor_in_every_stage`
/// (deleted with it in 0.2.0), which built one declaration and translated it in all three stages.
#[test]
fn a_promoted_resource_role_reads_the_same_wrapped_as_bare() {
    for role in FC_PROMOTED_RESOURCE_ROLES {
        let bare = vec![(*role).to_string()];
        let wrapped = vec!["function_constant".to_string(), (*role).to_string()];
        assert_eq!(
            resource_input_role(&wrapped),
            Some(*role),
            "a gated {role} keeps its role"
        );
        assert_eq!(
            resource_input_role(&wrapped),
            resource_input_role(&bare),
            "and reads the same as the ungated declaration"
        );
        assert_eq!(
            resource_input_role(&wrapped),
            primary_role(&wrapped),
            "and the same as the fragment decode's reader"
        );
    }
}

/// Every role the vertex output decode lowers is read past an `air.function_constant` wrapper.
///
/// Driven off `VERTEX_OUTPUT_ROLES`, which is the table the decode itself matches on -- a role
/// cannot be lowered without appearing here, so unlike a role list beside a `match` this cannot go
/// vacuous about a role the decode handles. That is the whole point of the table: the wrapper list
/// and the lowering used to be two spellings of one fact, and the wrapper list named no output role
/// at all.
#[test]
fn every_vertex_output_role_is_read_past_the_function_constant_wrapper() {
    for (role, _) in VERTEX_OUTPUT_ROLES {
        let bare = vec![(*role).to_string()];
        let wrapped = vec!["function_constant".to_string(), (*role).to_string()];
        let read = |strs: &[String]| {
            gated_role(strs, |role| vertex_output_kind(role).is_some()).map(str::to_string)
        };
        assert_eq!(
            read(&wrapped),
            Some((*role).to_string()),
            "a gated air.{role} return member declares air.{role}"
        );
        assert_eq!(read(&wrapped), read(&bare), "same as the bare declaration");
        assert_eq!(
            read(&wrapped),
            primary_role(&wrapped).map(str::to_string),
            "and same as the fragment decode's reader, which reads every wrapped role"
        );
    }
}

/// A gated BUILTIN is absent unless the module's own initializer turns it on; a gated VARYING is
/// present either way.
///
/// The asymmetry is [`AIR_SYSTEM_VALUE_ROLES`]', for the same reason.
/// Nothing writes a builtin whose predicate is not on, and declaring `Layer` or `ViewportIndex`
/// anyway puts a device capability in the module for a value nothing produces; four vertex sources
/// of a 14579-source corpus declare TWO gated `air.point_size` members under mutually exclusive
/// predicates, and promoting both emits two `PointSize` builtins, which is invalid. A varying is
/// kept because the stage consuming the same struct keeps it -- dropping it on one side renumbers
/// every later varying on that side only.
#[test]
fn a_gated_vertex_builtin_is_absent_and_a_gated_varying_is_not() {
    let module = |role: &str| {
        format!(
            r#"
@off = internal addrspace(2) global i8 0, align 1

!air.vertex = !{{!0}}
!0 = !{{ptr @vmain, !1, !9}}
!1 = !{{!2, !3, !5}}
!2 = !{{!"air.position", !"air.arg_type_name", !"float4"}}
!3 = !{{!"air.function_constant", !4, !"air.{role}", !"air.arg_type_name", !"float", !"air.arg_name", !"gated"}}
!4 = !{{ptr addrspace(2) @off, !"bool", !"gate"}}
!5 = !{{!"air.vertex_output", !"air.arg_type_name", !"float", !"air.arg_name", !"after"}}
!9 = !{{}}
"#
        )
    };
    for (role, kind) in VERTEX_OUTPUT_ROLES {
        // `air.position` is not a gated case: a vertex shader without a clip position is not a
        // vertex shader, and the decode gives member 0 the position whatever it says.
        if matches!(kind, VertexOutputKind::Position) {
            continue;
        }
        let m = parse_air_vertex_meta(&module(role)).0.unwrap();
        if kind.is_system_value() {
            assert_eq!(
                m.output_role_of(1),
                Some(&VertOutRole::FunctionConstantDisabled),
                "an off air.{role} builtin is absent"
            );
            assert_eq!(
                m.output_role_of(2),
                Some(&VertOutRole::Varying(0)),
                "and never consumed a varying location to begin with"
            );
        } else {
            assert_eq!(
                m.output_role_of(1),
                Some(&VertOutRole::Varying(0)),
                "a gated air.{role} keeps its slot"
            );
            assert_eq!(
                m.output_role_of(2),
                Some(&VertOutRole::Varying(1)),
                "and the member behind it keeps its own"
            );
        }
    }
}

/// Wrapped buffers require predicate evidence before becoming descriptor-backed resources.
#[test]
fn an_unpromoted_wrapped_role_stays_the_function_constant() {
    let bare_wrapper = vec!["function_constant".to_string()];
    assert_eq!(
        resource_input_role(&bare_wrapper),
        Some("function_constant")
    );

    let wrapped_buffer = vec!["function_constant".to_string(), "buffer".to_string()];
    assert_eq!(
        resource_input_role(&wrapped_buffer),
        Some("function_constant")
    );
}

/// [`declared_role`] reads the role behind an `air.function_constant` wrapper POSITIONALLY: the
/// wrapper is a marker plus the predicate it gates on, and the role is whatever follows.
///
/// Every case below is a node this reading has to tell apart from the others. A bare
/// function-constant parameter IS the function constant, and reading past its wrapper by SCANNING
/// lands on the node's own `air.arg_type_name`; asking [`resource_input_role`] instead answers `None`
/// for every gated role the emitter does not promote, which is exactly the set a caller needs this
/// for. The `air.amplification_id` row is a real corpus shape that was classified `Other`, bound to
/// a zero, and never reported.
#[test]
fn a_declared_role_is_read_past_the_wrapper_by_position() {
    for (node, expected) in [
        (
            r#"i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1"#,
            Some("buffer"),
        ),
        (
            r#"i32 0, !"air.function_constant", !9, !"air.texture", !"air.arg_type_name", !"texture2d<float, sample>""#,
            Some("texture"),
        ),
        (
            r#"i32 0, !"air.function_constant", !9, !"air.amplification_id""#,
            Some("amplification_id"),
        ),
        // A wrapper whose predicate is followed by a qualifier gates a parameter that has no role.
        (
            r#"i32 0, !"air.function_constant", !9, !"air.arg_type_name", !"uint""#,
            None,
        ),
        // ...as does one carrying no predicate at all.
        (
            r#"i32 0, !"air.function_constant", !"air.arg_type_name", !"uint""#,
            None,
        ),
        // `specialize_function_constant_metadata` rewrites a resolved-off wrapper to this, and a
        // declaration the pipeline leaves out declares no role to place.
        (
            r#"i32 0, !"air.function_constant_disabled", !9, !"air.texture""#,
            None,
        ),
        (r#"i32 0"#, None),
    ] {
        assert_eq!(declared_role(node).as_deref(), expected, "reading `{node}`");
    }
}

/// An unmodelled role behind a gate the module's own initializers do NOT resolve is still
/// unmodelled, at every declaration site that refuses one.
///
/// The two questions a function-constant gate answers are not complements. "May this declaration be
/// present" (`metadata_enabled_by_default`) reads an unresolved gate as absent; "did this variant
/// leave the declaration out" (`function_constant_gate_disabled`) reads it as unknown. Refusing an
/// unmodelled role is only excused by the second: a role the emitter cannot lower, behind a
/// predicate the module does not decide, still ends up bound to a zero the shader reads as a
/// barycentric coordinate or a sample mask.
///
/// The unresolved spelling here is the corpus's own -- the constructor stores a value the static
/// evaluator cannot fold -- so the global's `0` initializer is no longer evidence of anything.
#[test]
fn an_unmodelled_role_behind_an_unresolved_gate_is_still_refused() {
    // `store i8 0` folds; `store i8 %opaque` does not, and drops the global from the resolved set.
    let module = |stage: &str, list: &str, resolved: bool| {
        let stored = if resolved { "0" } else { "%opaque" };
        format!(
            r#"
@gate = internal addrspace(2) global i8 0, align 1

declare i8 @opaque()

define internal void @_GLOBAL__sub_I.metal() #0 section "air.static_init" {{
  %opaque = call i8 @opaque()
  store i8 {stored}, ptr addrspace(2) @gate, align 1
  ret void
}}

define void @E() {{
  ret void
}}

!air.{stage} = !{{!0}}
!0 = !{{ptr @E, !1, !2}}
{list}
!9 = !{{ptr addrspace(2) @gate, !"bool", !"gate"}}
"#
        )
    };
    // Each stage's own parameter list, plus the fragment return list -- the four places a declared
    // role is checked against the roles that stage models.
    let unmodelled_roles = |stage: &str, list: &str, resolved: bool| -> Vec<(u32, String)> {
        let ll = module(stage, list, resolved);
        match stage {
            "kernel" => {
                parse_air_kernel_meta(&ll)
                    .0
                    .expect("kernel metadata")
                    .unmodelled_input_params
            }
            "vertex" => {
                parse_air_vertex_meta(&ll)
                    .0
                    .expect("vertex metadata")
                    .unmodelled_input_params
            }
            _ => {
                let meta = parse_air_fragment_meta(&ll).0.expect("fragment metadata");
                if list.contains("air.vertex_output") {
                    meta.unmodelled_output_members
                } else {
                    meta.unmodelled_input_params
                }
            }
        }
    };
    let empty_first = "!1 = !{}";
    let sites = [
        (
            "kernel entry parameter",
            "kernel",
            format!(
                "{empty_first}\n!2 = !{{!3}}\n!3 = !{{i32 0, !\"air.function_constant\", !9, \
                 !\"air.render_target_array_index\"}}"
            ),
        ),
        (
            "fragment entry parameter",
            "fragment",
            format!(
                "{empty_first}\n!2 = !{{!3}}\n!3 = !{{i32 0, !\"air.function_constant\", !9, \
                 !\"air.simdgroup_index_in_threadgroup\"}}"
            ),
        ),
        (
            "vertex entry parameter",
            "vertex",
            format!(
                "{empty_first}\n!2 = !{{!3}}\n!3 = !{{i32 0, !\"air.function_constant\", !9, \
                 !\"air.point_coord\"}}"
            ),
        ),
        (
            "fragment return member",
            "fragment",
            "!1 = !{!3}\n!2 = !{}\n!3 = !{!\"air.function_constant\", !9, \
             !\"air.vertex_output\"}"
                .to_string(),
        ),
    ];
    for (site, stage, list) in sites {
        assert!(
            unmodelled_roles(stage, &list, true).is_empty(),
            "{site}: a gate this module's initializers drive to zero leaves the declaration out of \
             the variant, so there is nothing to refuse"
        );
        assert_eq!(
            unmodelled_roles(stage, &list, false).len(),
            1,
            "{site}: the gate is unresolved, so the role is still one this stage cannot lower"
        );
    }
}

/// A gated entry parameter that names a SYSTEM VALUE is classified as that system value unless the
/// module's own initializers turn the gate off.
///
/// The fragment decode always read the role past the wrapper and consulted the system-value list
/// only to drop the gated-OFF ones. The kernel and vertex decodes instead asked `resource_input_role`,
/// which collapses every role it does not PROMOTE back to the wrapper -- so a gated-ON
/// `air.thread_index_in_simdgroup` was classified `Other` and bound to a ZERO, in a module that
/// validated and reflected as though nothing were missing. All three stages now answer through
/// `present_system_value_role`, so the two halves of the rule cannot come apart per stage again.
#[test]
fn a_gated_on_system_value_is_the_system_value_at_every_stage() {
    let module = |stage: &str, list: &str, on: bool| {
        let value = u8::from(on);
        format!(
            r#"
@gate = internal addrspace(2) global i8 {value}, align 1

define void @E(i32 %sys) {{
  ret void
}}

!air.{stage} = !{{!0}}
!0 = !{{ptr @E, !1, !2}}
!1 = !{{}}
!2 = !{{!3}}
{list}
!9 = !{{ptr addrspace(2) @gate, !"bool", !"gate"}}
"#
        )
    };
    let node = |role: &str| {
        format!(
            "!3 = !{{i32 0, !\"air.function_constant\", !9, !\"air.{role}\", \
             !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"sys\"}}"
        )
    };

    for (role, expected) in [
        (
            "thread_index_in_simdgroup",
            KernRole::ExecutionGroup {
                fact: ExecutionGroupFact::ThreadIndexInGroup,
                lanes: 32,
            },
        ),
        (
            "thread_index_in_threadgroup",
            KernRole::ThreadIndexInThreadgroup,
        ),
    ] {
        let list = node(role);
        assert_eq!(
            parse_air_kernel_meta(&module("kernel", &list, true))
                .0
                .expect("kernel metadata")
                .role_of(0),
            Some(&expected),
            "a gated-ON air.{role} is the builtin the kernel reads"
        );
        assert_eq!(
            parse_air_kernel_meta(&module("kernel", &list, false))
                .0
                .expect("kernel metadata")
                .role_of(0),
            Some(&KernRole::Other),
            "...and a gated-OFF one is absent, so a zero is what Metal defines"
        );
    }

    for (role, expected) in [
        ("instance_id", VertRole::InstanceId),
        ("vertex_id", VertRole::VertexId),
    ] {
        let list = node(role);
        assert_eq!(
            parse_air_vertex_meta(&module("vertex", &list, true))
                .0
                .expect("vertex metadata")
                .role_of(0),
            Some(&expected),
            "a gated-ON air.{role} is the builtin the vertex shader reads"
        );
        assert_eq!(
            parse_air_vertex_meta(&module("vertex", &list, false))
                .0
                .expect("vertex metadata")
                .role_of(0),
            Some(&VertRole::Other),
            "...and a gated-OFF one is absent"
        );
    }

    // The fragment decode already answered both halves; it must keep answering them the same way.
    for (role, expected) in [
        ("front_facing", FragRole::FrontFacing),
        ("sample_id", FragRole::SampleId),
    ] {
        let list = node(role);
        assert_eq!(
            parse_air_fragment_meta(&module("fragment", &list, true))
                .0
                .expect("fragment metadata")
                .role_of(0),
            Some(&expected),
        );
        assert_eq!(
            parse_air_fragment_meta(&module("fragment", &list, false))
                .0
                .expect("fragment metadata")
                .role_of(0),
            Some(&FragRole::Other),
        );
    }
}
