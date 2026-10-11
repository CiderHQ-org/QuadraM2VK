//! The encoder's vertex amplification (`amplification.rs`).

use crate::amplification::{Amplification, VertexAmplificationViewMapping};
use crate::render::DrawProblem;
use metal2vulkan::bind::{VertexAmplification, ViewMapping};

/// The setter keeps the first `count` mappings, `nil` as none, and refuses fewer than the count;
/// the draw takes a count above a pipeline's maximum of 2 or more up to Metal's 8, as plain Metal
/// draws it, and refuses one above a maximum of 1, 0, and one past 8.
#[test]
fn an_amplification_is_held_to_its_mappings_and_the_pipeline() {
    let m = VertexAmplificationViewMapping::new;
    let given = [m(0, 2), m(0, 3), m(1, 0)];
    let two = Amplification::new(2, Some(&given)).unwrap();
    assert_eq!(
        two.bind_state(2),
        Ok(VertexAmplification::new(
            2,
            vec![ViewMapping::new(0, 2), ViewMapping::new(0, 3)]
        ))
    );
    assert_eq!(
        Amplification::new(2, None).unwrap().bind_state(4),
        Ok(VertexAmplification::new(2, Vec::new()))
    );
    assert_eq!(
        Amplification::default().bind_state(1),
        Ok(VertexAmplification::default())
    );
    assert_eq!(
        Amplification::new(3, Some(&given[..2])).unwrap_err(),
        DrawProblem::ViewMappings { count: 3, given: 2 }
    );
    for (count, max) in [(3, 2), (8, 2), (8, 4)] {
        assert_eq!(
            Amplification::new(count, None).unwrap().bind_state(max),
            Ok(VertexAmplification::new(count as u32, Vec::new())),
            "{count} {max}"
        );
    }
    for (count, max) in [(2, 1), (0, 2), (9, 2), (9, 8), (u64::from(u32::MAX) + 1, 8)] {
        assert_eq!(
            Amplification::new(count, None).unwrap().bind_state(max),
            Err(DrawProblem::AmplificationCount { count, max }),
            "{count} {max}"
        );
    }
}
