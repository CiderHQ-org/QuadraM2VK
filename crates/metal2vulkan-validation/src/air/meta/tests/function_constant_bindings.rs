use super::*;

/// A gated-off texture that STATES its Metal slot keeps its binding. `[[texture(0),
/// function_constant(a)]]` beside `[[texture(0), function_constant(!a)]]` is Metal's own spelling of
/// mutually exclusive typed alternatives: the slot is written down rather than summed, so it is this
/// argument's slot whichever alternative the pipeline enables.
#[test]
fn kernel_function_constant_texture_with_a_literal_location_is_bound_when_disabled_by_default() {
    let ll = r#"
@texture_enabled = internal addrspace(2) global i8 0, align 1

define void @K(ptr addrspace(1) %texture, ptr addrspace(2) %sampler) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @K, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", i32 40, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 15, i32 1, !"air.arg_type_name", !"sampler"}
!5 = !{ptr addrspace(2) @texture_enabled, !"bool", !"texture_enabled"}
"#;
    let meta = parse_air_kernel_meta(ll).0.expect("kernel metadata");
    assert_eq!(meta.role_of(0), Some(&KernRole::Texture(40)));
    assert_eq!(meta.role_of(1), Some(&KernRole::Sampler(15)));
}

/// ...and a gated-off texture whose slot is a RUNNING SUM the static initializer computes does not.
///
/// Metal compiles `[[texture(fc_expr)]]` into a sum over the arguments the pipeline enables, so for
/// an argument this variant leaves out the global holds wherever the sum stopped -- which is the
/// LIVE texture's own slot, as it is here. Binding the gated argument there decorates two
/// differently shaped images with one descriptor, so it must not be bound at all. 4851 corpus
/// `air.location_index` operands are spelled this way and every one of them is a global the static
/// initializer writes; none is a constant global like the literal case above.
#[test]
fn kernel_function_constant_texture_summed_onto_a_live_texture_slot_is_not_bound() {
    let ll = r#"
@live_enabled = internal addrspace(2) global i8 1, align 1
@gated_enabled = internal addrspace(2) global i8 0, align 1
@live_location = internal addrspace(2) global i32 0, align 4
@gated_location = internal addrspace(2) global i32 0, align 4

define internal void @_GLOBAL__sub_I_alternatives.metal() #0 section "air.static_init" {
  store i32 3, ptr addrspace(2) @live_location, align 4
  store i32 3, ptr addrspace(2) @gated_location, align 4
  ret void
}

define void @K(ptr addrspace(1) %live, ptr addrspace(1) %gated) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @K, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", ptr addrspace(2) @live_location, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>"}
!4 = !{i32 1, !"air.function_constant", !6, !"air.texture", !"air.location_index", ptr addrspace(2) @gated_location, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<float, sample>"}
!5 = !{ptr addrspace(2) @live_enabled, !"bool", !"live_enabled"}
!6 = !{ptr addrspace(2) @gated_enabled, !"bool", !"gated_enabled"}
"#;
    let meta = parse_air_kernel_meta(ll).0.expect("kernel metadata");
    assert_eq!(meta.role_of(0), Some(&KernRole::Texture(3)));
    // Not `Other`: the argument is a TEXTURE this variant leaves out, and saying so is what lets
    // the lowering tell its placeholder apart from a handle it merely lost.
    assert_eq!(meta.role_of(1), Some(&KernRole::VariantAbsentTexture));
}

#[test]
fn function_constant_texture_with_absent_location_is_not_bound() {
    let ll = r#"
@texture_location = internal addrspace(2) global i32 -1, align 4
@texture_enabled = internal addrspace(2) global i8 0, align 1

define void @K(ptr addrspace(1) %texture) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @K, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.function_constant", !4, !"air.texture", !"air.location_index", ptr addrspace(2) @texture_location, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>"}
!4 = !{ptr addrspace(2) @texture_enabled, !"bool", !"texture_enabled"}
"#;

    let meta = parse_air_kernel_meta(ll).0.expect("kernel metadata");
    assert_eq!(meta.role_of(0), Some(&KernRole::VariantAbsentTexture));
}

#[test]
fn kernel_function_constant_buffer_locations_preserve_shared_bindings() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @K, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.function_constant", !6, !"air.buffer", !"air.location_index", i32 29}
!4 = !{i32 1, !"air.function_constant", !7, !"air.buffer", !"air.location_index", i32 29}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 30}
!6 = !{ptr addrspace(2) @float_enabled, !"bool", !"float_enabled"}
!7 = !{ptr addrspace(2) @half_enabled, !"bool", !"half_enabled"}
"#;

    let meta = parse_air_kernel_meta(ll).0.expect("kernel metadata");
    assert_eq!(
        meta.function_constant_buffer_locations,
        HashMap::from([(0, 29), (1, 29)])
    );
}

/// Metal flattens `array_ref<void>` into consecutive buffer-table arguments beginning at the
/// literal slot; its global is the array extent, not a replacement binding for element zero.
///
/// This used to need a type-name special case in the buffer decode. It does not any more: the slot
/// is the FIRST `air.location_index` operand whatever the second one is, so the general positional
/// read gets it right and nothing keys on the spelling `array_ref<void>`. Kept as the case that
/// motivated the fix.
#[test]
fn device_buffer_array_uses_literal_base_instead_of_static_next_location() {
    let ll = r#"
@next_buffer = internal addrspace(2) global i32 1, align 4

!air.kernel = !{!0}
!0 = !{ptr @K, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, ptr addrspace(2) @next_buffer, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"array_ref<void>", !"air.arg_name", !"buffers"}
"#;
    let meta = parse_air_kernel_meta(ll).0.expect("kernel metadata");
    assert_eq!(meta.role_of(0), Some(&KernRole::Buffer(0)));
    assert_eq!(meta.buffer_type_name(0), Some("array_ref<void>"));
}

#[test]
fn fragment_function_constant_render_target_disabled_by_default() {
    let ll = r#"
!air.fragment = !{!0}
!0 = !{ptr @F, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{i32 7, !"air.function_constant", !5, !"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"half2"}
!4 = !{}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_1, !"bool", !"uses_coverage"}
"#;
    let m = parse_air_fragment_meta(ll).0.unwrap();
    assert_eq!(m.render_target_members.len(), 1);
    assert_eq!(m.render_target_members, vec![(0, 0)]);
    assert_eq!(m.render_target_type_name(1), None);
}

#[test]
fn fragment_function_constant_render_target_static_true_is_recognized() {
    let ll = r#"
@__metal_implicit_fc_pred_1 = internal unnamed_addr addrspace(2) global i8 1, align 1
!air.fragment = !{!0}
!0 = !{ptr @F, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{i32 7, !"air.function_constant", !5, !"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"half2"}
!4 = !{}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_1, !"bool", !"uses_coverage"}
"#;
    let m = parse_air_fragment_meta(ll).0.unwrap();
    assert_eq!(m.render_target_members.len(), 2);
    assert_eq!(m.render_target_members, vec![(0, 0), (1, 1)]);
    assert_eq!(m.render_target_type_name(1), Some("half2"));
}

/// A gated buffer the module's own static initializers drive ON is bound; the same declaration
/// behind a gate driven OFF, or behind one the module cannot resolve, stays absent. AIR shape from
/// `CC`-style `!"air.function_constant", !PRED, !"air.buffer"` arguments, whose predicate global is
/// written by the `air.static_init` constructor.
#[test]
fn a_gated_buffer_is_bound_only_when_the_module_drives_its_gate_on() {
    let ll = |stored: &str| {
        format!(
            r#"
@pred = internal addrspace(2) global i8 undef, align 1

define internal void @ctor() section "air.static_init" {{
  store i8 {stored}, ptr addrspace(2) @pred, align 1
  ret void
}}

define void @k(ptr addrspace(1) %optional) {{
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.function_constant", !4, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"optional"}}
!4 = !{{ptr addrspace(2) @pred, !"bool", !"kOptional"}}
"#
        )
    };

    let on = parse_air_kernel_meta(&ll("1")).0.expect("gate on");
    assert_eq!(on.role_of(0), Some(&KernRole::Buffer(7)));
    assert_eq!(on.buffer_address_space(0), Some(1));
    assert_eq!(on.buffer_type_size(0), Some(4));
    assert_eq!(on.buffer_type_name(0), Some("float"));

    let off = parse_air_kernel_meta(&ll("0")).0.expect("gate off");
    assert_eq!(off.role_of(0), Some(&KernRole::Other));

    // An initializer this evaluator cannot fold leaves the gate unresolved, and "may be present"
    // is not evidence of presence: the possibly-absent placeholder stays.
    let unresolved = parse_air_kernel_meta(&ll("%runtime"))
        .0
        .expect("gate unresolved");
    assert_eq!(unresolved.role_of(0), Some(&KernRole::Other));
}

/// The vertex decode answers the same declaration the same way. One gated buffer must not be a
/// binding on one stage and a Private placeholder on another.
#[test]
fn a_vertex_gated_buffer_follows_the_same_gate_evidence() {
    let ll = |stored: &str| {
        format!(
            r#"
@pred = internal addrspace(2) global i8 undef, align 1

define internal void @ctor() section "air.static_init" {{
  store i8 {stored}, ptr addrspace(2) @pred, align 1
  ret void
}}

define void @v(ptr addrspace(1) %optional) {{
  ret void
}}

!air.vertex = !{{!0}}
!0 = !{{ptr @v, !1, !2}}
!1 = !{{}}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.function_constant", !4, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"optional"}}
!4 = !{{ptr addrspace(2) @pred, !"bool", !"kOptional"}}
"#
        )
    };

    let on = parse_air_vertex_meta(&ll("1")).0.expect("gate on");
    assert_eq!(on.role_of(0), Some(&VertRole::Buffer(5)));

    let off = parse_air_vertex_meta(&ll("0")).0.expect("gate off");
    assert_eq!(off.role_of(0), Some(&VertRole::Other));
}

/// A gated buffer on a FRAGMENT entry keeps its binding whichever way the gate resolves, and that
/// is not the kernel and vertex answer above. The asymmetry is measured, not an oversight.
///
/// Making the fragment agree with the other two -- dropping the descriptor when the module's own
/// initializers drive the gate to zero -- changes 172 of the 14579 local corpus sources' SPIR-V
/// bytes and 175 of their reflections, for **490 buffer bindings dropped and none gained**. 132 of
/// those 490 are referenced by the module's own instructions: 1060 `OpLoad`s become `OpCopyObject`
/// of a null, in 100 modules. Nothing is rescued (zero status transitions) and the only gain is 10
/// fewer duplicate `(kind, set, binding)` slots.
///
/// The rule [`variant_texture_slot`] already applies says why the fragment is the right answer: a
/// gated-off argument declaring a LITERAL slot keeps it, because `[[buffer(0), function_constant(a)]]`
/// beside `[[buffer(0), function_constant(!a)]]` is Metal's own spelling of mutually exclusive
/// alternatives. Only a slot SUMMED through a function constant is fabricated when the argument is
/// absent -- and a census of all 933 gated fragment buffer declarations in the corpus finds **zero**
/// with a summed slot. The evidence that drops a texture slot does not exist for a buffer, and the
/// gate alone is not it: the seeded-to-zero `air.fc_initializer` default is what the module compiles
/// against, not what the pipeline binds.
#[test]
fn a_fragment_gated_buffer_keeps_its_literal_slot_whichever_way_the_gate_resolves() {
    let ll = |stored: &str| {
        format!(
            r#"
@pred = internal addrspace(2) global i8 undef, align 1

define internal void @ctor() section "air.static_init" {{
  store i8 {stored}, ptr addrspace(2) @pred, align 1
  ret void
}}

define void @f(ptr addrspace(1) %optional) {{
  ret void
}}

!air.fragment = !{{!0}}
!0 = !{{ptr @f, !1, !2}}
!1 = !{{!5}}
!5 = !{{!"air.render_target", i32 0, i32 0}}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.function_constant", !4, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"optional"}}
!4 = !{{ptr addrspace(2) @pred, !"bool", !"kOptional"}}
"#
        )
    };

    for stored in ["1", "0", "%runtime"] {
        let meta = parse_air_fragment_meta(&ll(stored))
            .0
            .expect("fragment meta");
        assert_eq!(
            meta.role_of(0),
            Some(&FragRole::Buffer(7)),
            "gate stored as {stored}"
        );
    }
}

/// The slot of an argument whose COUNT is a function-constant global is still the literal slot.
///
/// `array_ref<void>` is spelled `air.location_index, i32 N, ptr @extent`: a fixed buffer slot with
/// a runtime array extent. Resolving "the first global after the marker" through the static
/// initializers answers with the extent, binding the argument at a slot AIR never named. 439 corpus
/// nodes are that shape, 69 of them textures.
#[test]
fn a_function_constant_count_does_not_become_the_location_index() {
    let ll = FRAG_LL.replace(
        r#"!"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"texture2d<float, sample>""#,
        r#"!"air.location_index", i32 0, ptr addrspace(2) @extent.1, !"air.arg_type_name", !"texture2d<float, sample>""#,
    );
    let ll = format!(
        "@extent.1 = internal addrspace(2) global i32 9, align 4\n\
         define internal void @init() section \"air.static_init\" {{\n\
         entry:\n  store i32 9, ptr addrspace(2) @extent.1, align 4\n  ret void\n}}\n{ll}"
    );
    let meta = parse_air_fragment_meta(&ll).0.unwrap();
    assert_eq!(
        meta.role_of(2),
        Some(&FragRole::Texture(0)),
        "the texture binds at the slot AIR states, not at the value of its count global"
    );
}

/// A color attachment whose Location is a function constant is read the same way a resource slot
/// is: it is the FIRST operand after its marker, and the operands after it describe something else.
///
/// `!"air.render_target", ptr addrspace(2) @loc, i32 0` -- the trailing `i32 0` is the dual-source
/// index, `0` in all 3213 literal-form corpus declarations. A decode that answers with the next
/// `i32` it can find reports every function-constant render target at Location 0 whatever the
/// constant says, which for a multi-target shader is every output on one attachment. 103 of the
/// 3316 corpus `air.render_target` declarations spell the Location as a global.
#[test]
fn a_function_constant_render_target_location_is_the_operand_after_its_marker() {
    const FRAGMENT: &str = r#"
@_ZL32__metal_implicit_attr_int_expr_0.103 = internal addrspace(2) global i32 0, align 4

define internal void @_GLOBAL__sub_I_targets.metal() section "air.static_init" {
entry:
  store i32 SLOT, ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_0.103
  ret void
}

!air.fragment = !{!0}
!0 = !{ptr @F, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4", !"air.arg_name", !"color"}
!3 = !{!"air.render_target", ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_0.103, i32 0, !"air.arg_type_name", !"half4", !"air.arg_name", !"extra"}
!4 = !{}
"#;

    let resolved = FRAGMENT.replace("i32 SLOT", "i32 3");
    let meta = parse_air_fragment_meta(&resolved).0.unwrap();
    assert_eq!(
        meta.render_target_members,
        vec![(0, 0), (1, 3)],
        "the second member's Location is what its function constant initializes to, not the \
         dual-source index sitting after it"
    );

    // A global this module does not initialize at all is a Location the translator does not know.
    // The member ordinal stands in -- it is at least unique per member -- and the operands after
    // the slot still do not. Answering `0` here would put both outputs on attachment 0.
    let unresolved = resolved.replace(
        "@_ZL32__metal_implicit_attr_int_expr_0.103, i32 0, !\"air.arg_type_name\", !\"half4\", !\"air.arg_name\", !\"extra\"",
        "@__air_location_this_module_never_defines, i32 0, !\"air.arg_type_name\", !\"half4\", !\"air.arg_name\", !\"extra\"",
    );
    let meta = parse_air_fragment_meta(&unresolved).0.unwrap();
    assert_eq!(meta.render_target_members, vec![(0, 0), (1, 1)]);
}
