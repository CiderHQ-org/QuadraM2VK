//! `render.rs`: the faces a draw's stencil state commands set.

use crate::render::faces;
use ash::vk;

/// The faces and values of `front` and `back`'s commands, the faces as raw flags.
fn commands(front: u32, back: u32) -> Vec<(u32, u32)> {
    faces(front, back)
        .map(|(face, value)| (face.as_raw(), value))
        .collect()
}

/// Front and back faces of the same value take one command, of both faces; faces of different
/// values take one each, front first.
#[test]
fn equal_faces_take_one_stencil_command() {
    let raw = |face: vk::StencilFaceFlags| face.as_raw();
    assert_eq!(
        commands(0xff, 0xff),
        [(raw(vk::StencilFaceFlags::FRONT_AND_BACK), 0xff)]
    );
    assert_eq!(
        commands(1, 2),
        [
            (raw(vk::StencilFaceFlags::FRONT), 1),
            (raw(vk::StencilFaceFlags::BACK), 2)
        ]
    );
}
