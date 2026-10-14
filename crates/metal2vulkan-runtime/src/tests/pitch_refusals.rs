//! The kept refusals of `DRM_FORMAT_MOD_LINEAR` images at Metal's row pitch
//! (`texture/pitch_refusals.rs`): which later images a refusal answers, and how many are kept.

use super::super::image_plan::{plan_image, ImagePlan};
use super::super::texture::pitch_refusals::{PitchRefusals, PitchedLayout, KEPT};
use super::super::texture_descriptor::TextureDescriptor;
use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The image plan of a `BGRA8Unorm` texture of `width`x`height`, `layers` slices and `usage` on
/// the first committed profile.
fn plan(width: u64, height: u64, layers: u64, usage: TextureUsage) -> ImagePlan {
    let (_, desc, device) = committed().into_iter().next().unwrap();
    let layout = TextureDescriptor {
        pixel_format: PixelFormat::BGRA8Unorm,
        texture_type: if layers > 1 {
            TextureType::Type2DArray
        } else {
            TextureType::Type2D
        },
        width,
        height,
        array_length: layers,
        usage,
        ..TextureDescriptor::default()
    }
    .check(APPLE)
    .unwrap();
    plan_image(&layout, usage, false, &facts(&desc, &device)).unwrap()
}

/// A kept refusal answers the same image at the same row pitch only: another pitch of the same
/// power-of-two class, another extent or another usage is asked again (NVIDIA answers pitches of
/// one class both ways), and an image of several slices is never kept.
#[test]
fn a_refusal_answers_the_same_image_at_the_same_pitch_alone() {
    let read = TextureUsage::SHADER_READ;
    let tile = plan(1307, 512, 1, read);
    let refusals = PitchRefusals::default();
    let refused = PitchedLayout::of(&tile, 5248).unwrap();
    assert!(!refusals.refused(&refused));
    refusals.keep(refused.clone());
    assert!(refusals.refused(&refused));
    assert!(refusals.refused(&PitchedLayout::of(&tile, 5248).unwrap()));
    // 5376 = 42 x 128, the class of 5248 = 41 x 128.
    assert!(!refusals.refused(&PitchedLayout::of(&tile, 5376).unwrap()));
    let taller = plan(1307, 513, 1, read);
    assert!(!refusals.refused(&PitchedLayout::of(&taller, 5248).unwrap()));
    let rendered = plan(1307, 512, 1, read | TextureUsage::RENDER_TARGET);
    assert!(!refusals.refused(&PitchedLayout::of(&rendered, 5248).unwrap()));
    assert_eq!(PitchedLayout::of(&plan(1307, 512, 2, read), 5248), None);
}

/// At most [`KEPT`] refusals are kept: past them a refusal is not, and the driver is asked again.
#[test]
fn at_most_the_bound_of_refusals_is_kept() {
    let tile = plan(300, 200, 1, TextureUsage::SHADER_READ);
    let refusals = PitchRefusals::default();
    let pitch = |k: usize| 1216 + 16 * k as u64;
    for k in 0..KEPT {
        refusals.keep(PitchedLayout::of(&tile, pitch(k)).unwrap());
    }
    assert!((0..KEPT).all(|k| refusals.refused(&PitchedLayout::of(&tile, pitch(k)).unwrap())));
    let past = PitchedLayout::of(&tile, pitch(KEPT)).unwrap();
    refusals.keep(past.clone());
    assert!(!refusals.refused(&past));
}
