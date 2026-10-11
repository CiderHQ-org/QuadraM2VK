//! Labels and purgeable state (`attributes.rs`).

use crate::attributes::{set_purgeable, ResourceAttributes};
use crate::{Error, PurgeableState};

#[test]
fn purgeable_states_keep_metal_s_raw_values() {
    for (state, raw) in [
        (PurgeableState::KeepCurrent, 1),
        (PurgeableState::NonVolatile, 2),
        (PurgeableState::Volatile, 3),
        (PurgeableState::Empty, 4),
    ] {
        assert_eq!(state.raw(), raw);
        assert_eq!(PurgeableState::from_raw(raw), Ok(state));
    }
    for raw in [0, 5, u64::MAX] {
        assert!(matches!(
            PurgeableState::from_raw(raw),
            Err(Error::Invalid(_))
        ));
    }
}

#[test]
fn the_purgeable_state_answers_as_metal_s_did() {
    // The sequence the Metal probe sent to shared, managed and private buffers, and Metal's
    // answers on an M3 Max and an M4.
    use PurgeableState::*;
    let attributes = ResourceAttributes::new();
    let answers: Vec<PurgeableState> = [
        KeepCurrent,
        Volatile,
        KeepCurrent,
        Empty,
        KeepCurrent,
        NonVolatile,
        KeepCurrent,
    ]
    .into_iter()
    .map(|state| attributes.set_purgeable_state(state))
    .collect();
    assert_eq!(
        answers,
        [
            NonVolatile,
            NonVolatile,
            Volatile,
            Volatile,
            Empty,
            Empty,
            NonVolatile
        ]
    );
}

#[test]
fn a_volatile_resource_is_never_purged() {
    // Metal may purge a volatile resource (its state then answers Empty) and need not: the
    // runtime never does, so a query keeps answering Volatile.
    let mut state = PurgeableState::NonVolatile;
    (_, state) = set_purgeable(state, PurgeableState::Volatile);
    for _ in 0..3 {
        let (answer, next) = set_purgeable(state, PurgeableState::KeepCurrent);
        assert_eq!(
            (answer, next),
            (PurgeableState::Volatile, PurgeableState::Volatile)
        );
        state = next;
    }
}

#[test]
fn a_label_is_none_until_set_and_then_the_last_one() {
    let attributes = ResourceAttributes::new();
    assert_eq!(attributes.label(), None);
    attributes.set_label("first");
    attributes.set_label("guest buffer 7");
    assert_eq!(attributes.label().as_deref(), Some("guest buffer 7"));
}

#[test]
fn small_texture_allocations_are_suballocated_as_metal_s_were() {
    // Each texture shape the probes made on an Apple M3 Max and an M4, and whether Metal kept its
    // purgeable state (an allocation of its own, at least 16 KiB): `texture_allocation` must reach
    // `PURGEABLE_ALLOCATION` exactly for the shapes Metal kept.
    use crate::attributes::{texture_allocation, PURGEABLE_ALLOCATION};
    struct Shape {
        name: &'static str,
        block: [u32; 2],
        bytes: u64,
        extent: [u32; 3],
        volume: bool,
        levels: u32,
        layers: u32,
        samples: u32,
        kept: bool,
    }
    let plain = |name, bytes, w, h, kept| Shape {
        name,
        block: [1, 1],
        bytes,
        extent: [w, h, 1],
        volume: false,
        levels: 1,
        layers: 1,
        samples: 1,
        kept,
    };
    let mut shapes = vec![];
    for (side, kept) in [
        (8, false),
        (16, false),
        (24, false),
        (31, false),
        (32, false),
        (33, true),
        (40, true),
        (48, true),
        (56, true),
        (63, true),
        (64, true),
        (65, true),
        (128, true),
        (256, true),
        (1024, true),
    ] {
        shapes.push(plain("RGBA8Unorm square", 4, side, side, kept));
    }
    for (w, h, kept) in [
        (64, 16, false),
        (64, 63, true),
        (128, 32, true),
        (128, 31, true),
        (100, 41, true),
        (4096, 1, true),
        (4095, 1, true),
    ] {
        shapes.push(plain("RGBA8Unorm", 4, w, h, kept));
    }
    shapes.push(plain("R8Unorm 64x64", 1, 64, 64, false));
    shapes.push(plain("R8Unorm 128x128", 1, 128, 128, true));
    shapes.push(plain("R8Unorm 127x127", 1, 127, 127, true));
    shapes.push(plain("RGBA32Float 32x32", 16, 32, 32, true));
    shapes.push(plain("RGBA32Float 31x32", 16, 31, 32, true));
    shapes.push(plain("Depth32Float 64x64", 4, 64, 64, true));
    shapes.push(plain("Depth32Float 63x64", 4, 63, 64, true));
    for (side, levels, kept) in [(32, 6, false), (64, 7, true), (48, 6, true)] {
        shapes.push(Shape {
            levels,
            ..plain("RGBA8Unorm mipmapped", 4, side, side, kept)
        });
    }
    for (side, kept) in [(128, false), (256, true), (180, true)] {
        shapes.push(Shape {
            block: [4, 4],
            ..plain("BC1_RGBA", 8, side, side, kept)
        });
    }
    for (layers, kept) in [(4, true), (3, false)] {
        shapes.push(Shape {
            layers,
            ..plain("RGBA8Unorm 32x32 2D array", 4, 32, 32, kept)
        });
    }
    for (depth, kept) in [(16, true), (15, true)] {
        shapes.push(Shape {
            extent: [16, 16, depth],
            volume: true,
            ..plain("RGBA8Unorm 16x16 3D", 4, 16, 16, kept)
        });
    }
    for (side, kept) in [(32, true), (16, false)] {
        shapes.push(Shape {
            layers: 6,
            ..plain("RGBA8Unorm cube", 4, side, side, kept)
        });
    }
    for (w, h, kept) in [
        (32, 32, true),
        (31, 32, true),
        (16, 16, false),
        (256, 256, true),
    ] {
        shapes.push(Shape {
            samples: 4,
            ..plain("RGBA8Unorm 4x multisample", 4, w, h, kept)
        });
    }
    assert_eq!(shapes.len(), 45);
    for shape in shapes {
        let bytes = texture_allocation(
            shape.block,
            shape.bytes,
            shape.extent,
            shape.volume,
            shape.levels,
            shape.layers,
            shape.samples,
        );
        assert_eq!(
            bytes >= PURGEABLE_ALLOCATION,
            shape.kept,
            "{} {:?} x{} levels x{} layers x{} samples: {bytes}",
            shape.name,
            shape.extent,
            shape.levels,
            shape.layers,
            shape.samples
        );
    }
}
