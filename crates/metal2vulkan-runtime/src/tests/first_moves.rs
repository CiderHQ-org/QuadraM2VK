//! The work host calls stage for the device's next submission (`texture/first_move.rs`).

use crate::texture::first_move::{FirstMoves, TexelCopy};
use ash::vk;
use std::any::Any;
use std::sync::{Arc, Weak};

/// A host call's copy staged into an image makes a later CPU copy of its texels wait for the
/// submission that makes it; a decode alone staged (`TexelCopy::decode_only`: of what a copy on
/// the host or the staging queue wrote into a BC image, into its decoded twin) does not, as it
/// writes the twin alone (`FirstMoves::texels_wait`).
#[test]
fn a_staged_decode_alone_leaves_the_texels_to_cpu_copies() {
    let firsts = FirstMoves::default();
    let decoding: Arc<u32> = Arc::new(0);
    let copied: Arc<u32> = Arc::new(1);
    let keeper = |image: &Arc<u32>| -> Weak<dyn Any + Send + Sync> {
        let weak: Weak<u32> = Arc::downgrade(image);
        weak
    };
    let at = |image: &Arc<u32>| Arc::as_ptr(image).cast::<()>();
    // A decode's shape, its pass aside: no copy of texels.
    let alone = TexelCopy::new(vk::Buffer::null(), 0, Vec::new(), Box::new(()));
    assert!(!firsts.register_texels(keeper(&decoding), alone));
    let copy = (vk::Image::null(), vk::BufferImageCopy::default());
    let staged = TexelCopy::new(vk::Buffer::null(), 16, vec![copy], Box::new(()));
    assert!(!firsts.register_texels(keeper(&copied), staged));
    assert!(!firsts.texels_wait(at(&decoding)));
    assert!(firsts.texels_wait(at(&copied)));
}
