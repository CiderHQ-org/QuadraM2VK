//! A fragment case's authored depth/stencil state: the schema, the checks, the one state both
//! executors set from it, and the requirement it carries.

use super::*;
use crate::executor_contract::{case_requirements, DepthStencilTest, DrawDepthStencil};
use metal2vulkan::metal::{CompareFunction, StencilOperation, Winding};

/// A fragment case drawing into one 1x1 attachment holding `depth` and `stencil` aspects when
/// asked, under no authored state, observing its stencil (its depth without one).
fn fragment_case(depth: bool, stencil: bool) -> AuthoredCase {
    let mut case = example();
    case.stage = Stage::Fragment;
    case.dispatch = None;
    case.buffers = vec![];
    case.draw = Some(Draw {
        primitive: Primitive::Triangle,
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    case.depth_stencil = Some(DepthStencilResource {
        dimensions: [1, 1],
        initial_depth_b64: depth.then(|| "AACAPw==".into()),
        initial_stencil_b64: stencil.then(|| "qw==".into()),
        state: None,
    });
    case.output = if stencil {
        OutputSelection::Stencil {
            origin: [0, 0],
            dimensions: [1, 1],
        }
    } else {
        OutputSelection::Depth {
            origin: [0, 0],
            dimensions: [1, 1],
        }
    };
    case.case_id = case.computed_case_id().unwrap();
    case
}

fn face(compare: CompareFunctionLiteral, pass: StencilOperationLiteral) -> StencilFaceLiteral {
    StencilFaceLiteral {
        compare,
        stencil_failure: StencilOperationLiteral::Keep,
        depth_failure: StencilOperationLiteral::Keep,
        depth_stencil_pass: pass,
        read_mask: 0xFF,
        write_mask: 0xFF,
    }
}

/// A state that tests stencil `equal` against reference 3 on front faces and replaces on back
/// faces with reference 5, under counter-clockwise front faces, without a depth test.
fn stencil_state() -> DepthStencilStateLiteral {
    DepthStencilStateLiteral {
        depth_compare: CompareFunctionLiteral::Always,
        depth_write: false,
        front: StencilFaceLiteral {
            stencil_failure: StencilOperationLiteral::Invert,
            read_mask: 0x0F,
            write_mask: 0xF0,
            ..face(
                CompareFunctionLiteral::Equal,
                StencilOperationLiteral::IncrementWrap,
            )
        },
        back: face(
            CompareFunctionLiteral::Always,
            StencilOperationLiteral::Replace,
        ),
        front_reference: 3,
        back_reference: 5,
        front_facing_winding: WindingLiteral::CounterClockwise,
    }
}

fn errors(case: &AuthoredCase) -> Vec<String> {
    let mut case = case.clone();
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().err().unwrap_or_default()
}

/// The literal's spelling: snake_case names, and the field is absent from a case that authors
/// none, so every case written before it keeps its identity.
#[test]
fn the_state_round_trips_and_is_absent_when_unauthored() {
    let mut case = fragment_case(false, true);
    let unauthored = serde_json::to_value(&case).unwrap();
    assert!(unauthored["depth_stencil"].get("state").is_none());
    case.depth_stencil.as_mut().unwrap().state = Some(stencil_state());
    let json = serde_json::to_value(&case).unwrap();
    let state = &json["depth_stencil"]["state"];
    assert_eq!(state["front"]["compare"], "equal");
    assert_eq!(state["front"]["depth_stencil_pass"], "increment_wrap");
    assert_eq!(state["front_facing_winding"], "counter_clockwise");
    let back: AuthoredCase = serde_json::from_value(json).unwrap();
    assert_eq!(back, case);
    assert_ne!(
        case.computed_case_id().unwrap(),
        fragment_case(false, true).computed_case_id().unwrap()
    );
}

/// Masks and references are 8-bit, and a state tests or writes only aspects the attachment
/// authors.
#[test]
fn the_checks_hold_the_state_to_8_bits_and_to_its_authored_aspects() {
    let mut case = fragment_case(false, true);
    case.depth_stencil.as_mut().unwrap().state = Some(stencil_state());
    assert!(errors(&case).is_empty(), "{:?}", errors(&case));

    let mut wide = stencil_state();
    wide.back.write_mask = 0x1FF;
    wide.front_reference = 0x100;
    case.depth_stencil.as_mut().unwrap().state = Some(wide);
    let found = errors(&case);
    assert!(
        found.iter().any(|e| e.contains("back.write_mask 0x1ff")),
        "{found:?}"
    );
    assert!(
        found.iter().any(|e| e.contains("front_reference 0x100")),
        "{found:?}"
    );

    // A stencil test without an authored stencil, and a depth write without an authored depth.
    let mut case = fragment_case(true, false);
    case.depth_stencil.as_mut().unwrap().state = Some(stencil_state());
    assert!(errors(&case)
        .iter()
        .any(|e| e.contains("without an authored stencil")));
    let mut case = fragment_case(false, true);
    case.depth_stencil.as_mut().unwrap().state = Some(DepthStencilStateLiteral {
        depth_write: true,
        ..stencil_state()
    });
    assert!(errors(&case)
        .iter()
        .any(|e| e.contains("without an authored depth")));
    // A state that tests neither aspect needs neither.
    let identity = DepthStencilStateLiteral {
        front: face(
            CompareFunctionLiteral::Always,
            StencilOperationLiteral::Keep,
        ),
        back: face(
            CompareFunctionLiteral::Always,
            StencilOperationLiteral::Keep,
        ),
        ..stencil_state()
    };
    let mut case = fragment_case(true, false);
    case.depth_stencil.as_mut().unwrap().state = Some(identity);
    assert!(errors(&case).is_empty(), "{:?}", errors(&case));
}

/// The authored state is the descriptor, references, and winding as written; the derived state of
/// a `[[stencil]]` output is the authored state that always passes and replaces, and Metal's
/// defaults otherwise.
#[test]
fn both_executors_set_the_authored_state_as_written() {
    let draw = DrawDepthStencil::authored(&stencil_state());
    let front = draw.descriptor.front_face_stencil;
    assert_eq!(front.stencil_compare_function, CompareFunction::Equal);
    assert_eq!(front.stencil_failure_operation, StencilOperation::Invert);
    assert_eq!(
        front.depth_stencil_pass_operation,
        StencilOperation::IncrementWrap
    );
    assert_eq!((front.read_mask, front.write_mask), (0x0F, 0xF0));
    assert_eq!(
        draw.descriptor
            .back_face_stencil
            .depth_stencil_pass_operation,
        StencilOperation::Replace
    );
    assert_eq!((draw.front_reference, draw.back_reference), (3, 5));
    assert_eq!(draw.winding, Winding::CounterClockwise);
    assert!(draw.tests_stencil() && !draw.tests_depth());

    let replace = DepthStencilStateLiteral {
        depth_compare: CompareFunctionLiteral::Always,
        depth_write: false,
        front: StencilFaceLiteral {
            read_mask: u32::MAX,
            write_mask: u32::MAX,
            ..face(
                CompareFunctionLiteral::Always,
                StencilOperationLiteral::Replace,
            )
        },
        back: StencilFaceLiteral {
            read_mask: u32::MAX,
            write_mask: u32::MAX,
            ..face(
                CompareFunctionLiteral::Always,
                StencilOperationLiteral::Replace,
            )
        },
        front_reference: 0,
        back_reference: 0,
        front_facing_winding: WindingLiteral::Clockwise,
    };
    assert_eq!(
        DrawDepthStencil::derived(DepthStencilTest {
            depth: None,
            stencil: true,
        }),
        DrawDepthStencil::authored(&replace)
    );
    let default = DrawDepthStencil::derived(DepthStencilTest::default());
    assert_eq!(default, DrawDepthStencil::default());
    assert!(!default.tests_stencil() && !default.tests_depth());
}

/// An authored state is no executor's gap: both executors set it.
#[test]
fn an_authored_state_stops_no_executor() {
    let mut case = fragment_case(false, true);
    assert!(case_requirements(&case).is_empty());
    case.depth_stencil.as_mut().unwrap().state = Some(stencil_state());
    assert!(case_requirements(&case).is_empty());
}
