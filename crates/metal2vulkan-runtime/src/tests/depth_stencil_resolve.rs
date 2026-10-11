//! The depth/stencil resolve draws' plan: which draws run, in which order, over which bindings, for
//! each set of aspects drawn after the pass.

use super::super::depth_stencil_resolve::Drawn;
use ash::vk;
use metal2vulkan::bind::{DepthStencilResolveModules, ResolveWrite, StencilBit};
use metal2vulkan::vk::ImageAspectFlags;

#[test]
fn drawn_aspects_name_their_draws() {
    let depth = ImageAspectFlags::DEPTH;
    let stencil = ImageAspectFlags::STENCIL;
    assert_eq!(Drawn::of(ImageAspectFlags::empty()), None);
    assert_eq!(Drawn::of(ImageAspectFlags::COLOR), None);
    assert_eq!(Drawn::of(depth), Some(Drawn::Depth));
    assert_eq!(Drawn::of(stencil), Some(Drawn::Stencil));
    assert_eq!(Drawn::of(depth | stencil), Some(Drawn::Both));
}

/// The depth is drawn first (the stencil planes run with the depth test off), each plane once in
/// bit order; a depth drawn alone binds the depth aspect only, so a depth-only format needs no
/// stencil view.
#[test]
fn the_depth_draws_before_the_stencil_planes() {
    let planes: Vec<_> = StencilBit::ALL
        .into_iter()
        .map(ResolveWrite::StencilPlane)
        .collect();
    assert_eq!(Drawn::Depth.writes(), vec![ResolveWrite::Depth]);
    assert_eq!(Drawn::Stencil.writes(), planes);
    let mut both = vec![ResolveWrite::Depth];
    both.extend(planes);
    assert_eq!(Drawn::Both.writes(), both);
    let sampled = vk::DescriptorType::SAMPLED_IMAGE;
    assert_eq!(DepthStencilResolveModules::DEPTH_BINDING, 0);
    assert_eq!(DepthStencilResolveModules::STENCIL_BINDING, 1);
    // `vk::DescriptorType` has no `Debug` here, so the sets compare by `assert!`.
    assert!(Drawn::Depth.set() == [sampled]);
    assert!(Drawn::Stencil.set() == [sampled; 2]);
    assert!(Drawn::Both.set() == [sampled; 2]);
}
