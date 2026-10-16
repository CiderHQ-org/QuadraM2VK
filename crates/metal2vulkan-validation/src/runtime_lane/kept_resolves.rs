//! Device tests of the exact color resolve's kept objects (the runtime's `color_resolve.rs`,
//! `KeptResolves`) on the device the cfg selects, under the validation layer, through the resolve
//! lane's fixture and helpers (`resolves.rs`): a pass resolving the same views as the last takes the
//! descriptor sets and framebuffers that pass made, and still resolves its own samples; a pass of a
//! multisampled texture made after the last one dropped makes its own, whatever handles the driver
//! gives the new texture's views. Where the device resolves by its own average, or reads a copy of
//! the samples (its image takes no input attachment usage for free), no pass is kept.

use super::buffer_with;
use super::resolves::{color_pass, lane, quiet, Lane, FULL, MASKED};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{BlitOption, ClearColor, StoreAction, Texture, WorkCounts};

const RGBA8: PixelFormat = PixelFormat::RGBA8Unorm;

/// Resolve `msaa` cleared to `clear` (every channel), sample 0 then drawn white, into `target`;
/// the device's counts after.
fn resolve(lane: &Lane, msaa: &Texture, target: &Texture, clear: f64) -> WorkCounts {
    let pipeline = lane.pipeline((FULL, MASKED), &[RGBA8], (None, None), 4);
    let mask = buffer_with(&lane.device, &1u32.to_le_bytes()).unwrap();
    let mut pass = color_pass(msaa, Some(target), StoreAction::MultisampleResolve, (0, 0));
    pass.color_attachments[0].clear_color = ClearColor::new(clear, clear, clear, clear);
    lane.run(&pass, |l, e, h| {
        e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
        l.draw(e, h, &pipeline, 0.5);
    })
    .unwrap();
    lane.device.work_counts()
}

/// Whether every channel of every texel of `target` is within a unit of the mean of white and
/// three samples of `clear`.
fn resolved(lane: &Lane, target: &Texture, clear: f64) -> bool {
    let mean = (255.0 + 3.0 * (clear * 255.0).round()) / 4.0;
    let texels = lane.read(target, (0, 0), 4, BlitOption::None);
    texels.iter().all(|&v| (f64::from(v) - mean).abs() <= 1.0)
}

#[test]
fn a_pass_of_the_same_views_takes_the_kept_resolve_objects_and_a_new_texture_s_makes_its_own() {
    let (messages, _cache, lane) = lane();
    let target = lane.single(RGBA8);
    let msaa = lane.msaa(RGBA8);
    let before = lane.device.work_counts();
    let first = resolve(&lane, &msaa, &target, 0.0);
    assert!(resolved(&lane, &target, 0.0), "the first pass");
    let made = first.color_resolve_passes_made - before.color_resolve_passes_made;
    if made == 0 {
        // The device resolves RGBA8 at 4 samples by its own average (Metal's, on MoltenVK).
        quiet(lane, &messages);
        return;
    }
    assert_eq!(made, 1);
    let second = resolve(&lane, &msaa, &target, 1.0);
    assert!(resolved(&lane, &target, 1.0), "the second pass");
    let kept = second.color_resolve_passes_kept - first.color_resolve_passes_kept;
    let made = second.color_resolve_passes_made - first.color_resolve_passes_made;
    assert_eq!(kept + made, 1);
    if kept == 0 {
        // The device's 4x image takes no input attachment usage without a cost to its memory, so
        // the resolve reads a copy of its samples, through views a pass makes for itself: no pass
        // is kept.
        drop(msaa);
        quiet(lane, &messages);
        return;
    }
    drop(msaa);
    let msaa = lane.msaa(RGBA8);
    let third = resolve(&lane, &msaa, &target, 0.0);
    assert!(resolved(&lane, &target, 0.0), "the new texture's pass");
    assert_eq!(
        (
            third.color_resolve_passes_made - second.color_resolve_passes_made,
            third.color_resolve_passes_kept - second.color_resolve_passes_kept,
        ),
        (1, 0),
        "the new texture's pass took the dropped texture's objects"
    );
    drop(msaa);
    quiet(lane, &messages);
}
