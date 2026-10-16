//! The registry of new textures that read zero where Metal reads its pattern
//! (`fresh_unwritten.rs`): a read that reaches an unwritten texel is announced once, writes that
//! together cover every texel end the texture's entry, each aspect is written apart, and a
//! release forgets it.

use crate::fresh_unwritten::{FreshKind, FreshUnwritten, Notice};
use crate::resource::ResourceKey;
use crate::sparse::emulated::Written;
use crate::texture_descriptor::TextureLayout;
use ash::vk::ImageAspectFlags;
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::vk::Format;
use std::ops::Range;

const DEPTH: ImageAspectFlags = ImageAspectFlags::DEPTH;
const STENCIL: ImageAspectFlags = ImageAspectFlags::STENCIL;

/// A 4x `Depth32Float` 16² texture's notice.
fn notice() -> Notice {
    Notice {
        layout: TextureLayout {
            texture_type: TextureType::Type2DMultisample,
            pixel_format: PixelFormat::Depth32Float,
            extent: [16, 16, 1],
            levels: 1,
            layers: 1,
            samples: 4,
        },
        format: Format::D32_SFLOAT,
        kind: FreshKind::Zeroed,
    }
}

/// The box `[x, y, z]` of `size` at `origin` in `level`'s `layers`.
fn texels(level: u32, layers: Range<u32>, origin: [u32; 3], size: [u32; 3]) -> Written {
    Written {
        level,
        layers,
        origin,
        size,
    }
}

/// The whole of `level`'s `layers` (the tracker clips a box to the level).
fn whole(level: u32, layers: Range<u32>) -> Written {
    texels(level, layers, [0; 3], [u32::MAX; 3])
}

/// A read that reaches an unwritten texel is announced, once; one of a written subresource is
/// not, and outside the pattern's levels nothing is noted.
#[test]
fn a_read_before_a_whole_write_is_announced_once() {
    let fresh = FreshUnwritten::default();
    assert!(!fresh.any());
    let key = ResourceKey::new();
    fresh.note(key, 2, 3, DEPTH, [16, 16, 1], notice());
    assert!(fresh.any());
    fresh.written(key, DEPTH, &whole(0, 0..3));
    assert!(
        fresh.observed(key, DEPTH, 0..1, 0..3).is_none(),
        "level 0 is written"
    );
    assert!(
        fresh.observed(key, DEPTH, 2..4, 0..3).is_none(),
        "past the pattern's levels"
    );
    assert!(
        fresh.observed(key, DEPTH, 1..2, 2..3).is_some(),
        "level 1 is not"
    );
    assert!(!fresh.any(), "announced: no longer noted");
    assert!(fresh.observed(key, DEPTH, 1..2, 0..3).is_none());
}

/// Writing every level's every layer ends the entry; a release forgets one.
#[test]
fn a_whole_write_or_a_release_ends_the_entry() {
    let fresh = FreshUnwritten::default();
    let (written, released) = (ResourceKey::new(), ResourceKey::new());
    fresh.note(written, 1, 2, DEPTH, [16, 16, 1], notice());
    fresh.note(released, 1, 1, DEPTH, [16, 16, 1], notice());
    fresh.written(written, DEPTH, &whole(0, 0..1));
    assert!(fresh.observed(written, DEPTH, 0..1, 1..2).is_some());
    fresh.note(written, 1, 2, DEPTH, [16, 16, 1], notice());
    fresh.written(written, DEPTH, &whole(0, 0..1));
    fresh.written(written, DEPTH, &whole(0, 1..2));
    assert!(fresh.observed(written, DEPTH, 0..1, 0..2).is_none());
    assert!(fresh.any(), "the other texture stays");
    fresh.forget(released);
    assert!(!fresh.any());
    assert!(fresh.observed(released, DEPTH, 0..1, 0..1).is_none());
}

/// Parts that together cover a level write it, in any order (an upload in pieces): the entry
/// goes with the last, and no read before it of a written part is announced.
#[test]
fn parts_that_together_cover_a_level_write_it() {
    let fresh = FreshUnwritten::default();
    let quarters = [[32, 0], [0, 32], [32, 32], [0, 0]];
    let key = ResourceKey::new();
    fresh.note(key, 1, 1, DEPTH, [64, 64, 1], notice());
    for (n, [x, y]) in quarters.into_iter().enumerate() {
        assert!(fresh.any(), "{n} quarters written");
        fresh.written(key, DEPTH, &texels(0, 0..1, [x, y, 0], [32, 32, 1]));
        let read = texels(0, 0..1, [x, y, 0], [32, 32, 1]);
        assert!(fresh.observed_within(key, DEPTH, &read).is_none());
    }
    assert!(!fresh.any(), "the four quarters wrote the level");
    // Rows of one texel, as a buffer copied in by rows.
    let key = ResourceKey::new();
    fresh.note(key, 1, 2, DEPTH, [54, 54, 1], notice());
    for row in (0..54).rev() {
        fresh.written(key, DEPTH, &texels(0, 0..2, [0, row, 0], [54, 1, 1]));
    }
    assert!(!fresh.any(), "the rows wrote both layers");
}

/// A read is announced only where it reaches a texel no write covered.
#[test]
fn a_read_is_announced_where_it_reaches_an_unwritten_texel() {
    let fresh = FreshUnwritten::default();
    let left = texels(0, 0..1, [0; 3], [8, 16, 1]);
    let right = texels(0, 0..1, [8, 0, 0], [8, 16, 1]);
    let reads = [
        (left.clone(), false),
        (right, true),
        (texels(0, 0..1, [7, 4, 0], [2, 2, 1]), true),
        (texels(0, 0..1, [3, 4, 0], [5, 12, 1]), false),
    ];
    for (read, announced) in reads {
        let key = ResourceKey::new();
        fresh.note(key, 1, 1, DEPTH, [16, 16, 1], notice());
        fresh.written(key, DEPTH, &left);
        let notice = fresh.observed_within(key, DEPTH, &read);
        assert_eq!(notice.is_some(), announced, "{read:?}");
        assert!(
            fresh.observed(key, DEPTH, 0..1, 0..1).is_some() != announced,
            "a texture announced is no longer noted"
        );
    }
}

/// Each aspect is written apart: a copy of a depth/stencil texture's depth leaves its stencil
/// unwritten.
#[test]
fn each_aspect_is_written_apart() {
    let fresh = FreshUnwritten::default();
    let key = ResourceKey::new();
    fresh.note(key, 1, 1, DEPTH | STENCIL, [16, 16, 1], notice());
    fresh.written(key, DEPTH, &whole(0, 0..1));
    assert!(fresh.observed(key, DEPTH, 0..1, 0..1).is_none());
    assert!(fresh.observed(key, STENCIL, 0..1, 0..1).is_some());
    let key = ResourceKey::new();
    fresh.note(key, 1, 1, DEPTH | STENCIL, [16, 16, 1], notice());
    fresh.written(key, STENCIL, &whole(0, 0..1));
    fresh.written(key, DEPTH, &whole(0, 0..1));
    assert!(!fresh.any(), "both aspects written");
}

/// A 3D image's depth planes are its boxes' third axis: planes written apart write the level.
#[test]
fn a_volume_is_written_by_its_planes() {
    let fresh = FreshUnwritten::default();
    let colour = ImageAspectFlags::COLOR;
    let key = ResourceKey::new();
    fresh.note(key, 1, 1, colour, [16, 16, 8], notice());
    fresh.written(key, colour, &texels(0, 0..1, [0; 3], [16, 16, 4]));
    let far = texels(0, 0..1, [0, 0, 4], [16, 16, 4]);
    assert!(fresh.observed_within(key, colour, &far).is_some());
    let key = ResourceKey::new();
    fresh.note(key, 1, 1, colour, [16, 16, 8], notice());
    for plane in 0..8 {
        fresh.written(key, colour, &texels(0, 0..1, [0, 0, plane], [16, 16, 1]));
    }
    assert!(!fresh.any());
}

/// A small generator of boxes, the same each run.
struct Boxes(u64);

impl Boxes {
    fn below(&mut self, n: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % u64::from(n)) as u32
    }

    /// A box inside `[width, height]`, of a side up to `most`.
    fn next(&mut self, [width, height]: [u32; 2], most: u32) -> Written {
        let size = [1 + self.below(most), 1 + self.below(most)];
        let origin = [self.below(width), self.below(height)];
        texels(0, 0..1, [origin[0], origin[1], 0], [size[0], size[1], 1])
    }
}

/// Against a map of every texel, after writes of boxes at random: a texel no write covered is
/// always announced (never a silent read), and while the record stays within its bound (32
/// boxes; these sequences of boxes wider than a texel keep 22 at most) a written texel never is,
/// and the entry goes once every texel is written (seed 6's 250 boxes cover the level); past the
/// bound (single texels at random, a checkerboard) the record still holds every unwritten texel.
#[test]
fn the_record_holds_every_unwritten_texel() {
    let extent = [13u32, 11];
    let sequences = [
        (1u64, 12, 6),
        (2, 30, 4),
        (3, 6, 13),
        (4, 40, 3),
        (6, 250, 7),
        (5, 90, 1),
    ];
    let mut covered = 0;
    for (seed, writes, most) in sequences {
        let mut boxes = Boxes(seed);
        let sequence: Vec<Written> = (0..writes).map(|_| boxes.next(extent, most)).collect();
        let mut map = vec![false; (extent[0] * extent[1]) as usize];
        for b in &sequence {
            for y in b.origin[1]..(b.origin[1] + b.size[1]).min(extent[1]) {
                for x in b.origin[0]..(b.origin[0] + b.size[0]).min(extent[0]) {
                    map[(y * extent[0] + x) as usize] = true;
                }
            }
        }
        let replay = |fresh: &FreshUnwritten| {
            let key = ResourceKey::new();
            fresh.note(key, 1, 1, DEPTH, [extent[0], extent[1], 1], notice());
            for b in &sequence {
                fresh.written(key, DEPTH, b);
            }
            key
        };
        let fresh = FreshUnwritten::default();
        for y in 0..extent[1] {
            for x in 0..extent[0] {
                let key = replay(&fresh);
                let texel = texels(0, 0..1, [x, y, 0], [1, 1, 1]);
                let announced = fresh.observed_within(key, DEPTH, &texel).is_some();
                let written = map[(y * extent[0] + x) as usize];
                assert!(
                    announced || written,
                    "seed {seed}: ({x}, {y}) read silently"
                );
                if most > 1 {
                    assert!(
                        !announced || !written,
                        "seed {seed}: ({x}, {y}) was written"
                    );
                }
                fresh.forget(key);
            }
        }
        let key = replay(&fresh);
        let all = map.iter().all(|&w| w);
        covered += usize::from(all);
        if most > 1 {
            assert_eq!(
                fresh.any(),
                !all,
                "seed {seed}: the entry goes as the level is written"
            );
        }
        fresh.forget(key);
    }
    assert_eq!(covered, 1, "one sequence covers the level");
    // A checkerboard of single texels splits the unwritten texels past the bound.
    let fresh = FreshUnwritten::default();
    let key = ResourceKey::new();
    fresh.note(key, 1, 1, DEPTH, [16, 16, 1], notice());
    for y in 0..16 {
        for x in (y % 2..16).step_by(2) {
            fresh.written(key, DEPTH, &texels(0, 0..1, [x, y, 0], [1, 1, 1]));
        }
    }
    let white = texels(0, 0..1, [1, 0, 0], [1, 1, 1]);
    assert!(fresh.observed_within(key, DEPTH, &white).is_some());
}

/// A new render target's signed NaN depth pattern (`FreshKind::SignedDepth`) is observed by no
/// command's read, only by a pass that loads an unwritten texel of its depth, once; a load of the
/// written texels alone does not observe it. A pattern zeroed is not a signed depth's: a pass's
/// load of it reports nothing here, and a command's read announces it as before.
#[test]
fn a_signed_depth_pattern_is_observed_by_a_loading_pass_alone() {
    let fresh = FreshUnwritten::default();
    let key = ResourceKey::new();
    let signed = Notice {
        layout: TextureLayout {
            texture_type: TextureType::Type2D,
            samples: 1,
            ..notice().layout
        },
        kind: FreshKind::SignedDepth,
        ..notice()
    };
    fresh.note(key, 1, 1, DEPTH, [16, 16, 1], signed);
    assert!(
        fresh.observed(key, DEPTH, 0..1, 0..1).is_none(),
        "a command's read"
    );
    let all = whole(0, 0..1);
    assert!(
        fresh.observed_within(key, DEPTH, &all).is_none(),
        "a command's read of a box"
    );
    assert!(fresh.any(), "still noted");
    let left = texels(0, 0..1, [0; 3], [8, 16, 1]);
    fresh.written(key, DEPTH, &left);
    assert!(!fresh.loaded_signed_depth(key, &left), "the written half");
    assert!(fresh.loaded_signed_depth(key, &all), "an unwritten texel");
    assert!(!fresh.any(), "observed: no longer noted");
    assert!(!fresh.loaded_signed_depth(key, &all), "once");
    let zeroed = ResourceKey::new();
    fresh.note(zeroed, 1, 1, DEPTH, [16, 16, 1], notice());
    assert!(!fresh.loaded_signed_depth(zeroed, &all), "a zeroed pattern");
    assert!(
        fresh.observed(zeroed, DEPTH, 0..1, 0..1).is_some(),
        "a zeroed pattern's read"
    );
}
