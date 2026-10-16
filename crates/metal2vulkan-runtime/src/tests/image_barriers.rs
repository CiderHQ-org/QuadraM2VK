//! Every image barrier the runtime records names, in its destination, the accesses of an image
//! the recording queue family's commands make: never `MEMORY_READ` or `HOST_READ`, and never an
//! access the family's stages do not support (`image_barrier.rs`).

use crate::image_barrier::{image_barrier, ImageDst};
use crate::texture::first_move::FirstMove;
use crate::texture::wrapped::Transition;
use ash::vk;
use metal2vulkan::vk::{Format, ImageAspectFlags, ImageLayout};

/// What MoltenVK takes as a host read to come after a barrier into `GENERAL`.
const HOST_OR_ANY_READ: vk::AccessFlags = vk::AccessFlags::from_raw(
    vk::AccessFlags::MEMORY_READ.as_raw() | vk::AccessFlags::HOST_READ.as_raw(),
);

/// The kinds of queue family a barrier may be recorded for: transfers alone (a copy engine, with
/// sparse binding as NVIDIA's), compute without graphics, and graphics.
const FAMILIES: [vk::QueueFlags; 6] = [
    vk::QueueFlags::TRANSFER,
    vk::QueueFlags::from_raw(
        vk::QueueFlags::TRANSFER.as_raw() | vk::QueueFlags::SPARSE_BINDING.as_raw(),
    ),
    vk::QueueFlags::COMPUTE,
    vk::QueueFlags::from_raw(vk::QueueFlags::COMPUTE.as_raw() | vk::QueueFlags::TRANSFER.as_raw()),
    vk::QueueFlags::from_raw(vk::QueueFlags::GRAPHICS.as_raw() | vk::QueueFlags::COMPUTE.as_raw()),
    vk::QueueFlags::from_raw(
        vk::QueueFlags::GRAPHICS.as_raw()
            | vk::QueueFlags::COMPUTE.as_raw()
            | vk::QueueFlags::TRANSFER.as_raw()
            | vk::QueueFlags::SPARSE_BINDING.as_raw(),
    ),
];

/// What the stages of a family must hold for an access to be named after `ALL_COMMANDS` there.
#[derive(Clone, Copy)]
enum Needs {
    /// Any family's (`MEMORY_READ`, `MEMORY_WRITE`).
    Any,
    /// The copy stages, of every family of transfers, compute or graphics.
    Transfer,
    /// The compute shader stage, or a graphics family's shader stages, or the draw-indirect stage.
    Compute,
    /// The vertex input, fragment and attachment stages of a graphics family.
    Graphics,
    /// The host stage, which no family's `ALL_COMMANDS` holds.
    Host,
}

/// Each access a pipeline stage makes, by the stages it needs (the Vulkan specification's
/// "Supported access types" table).
const ACCESS_STAGES: [(vk::AccessFlags, Needs); 17] = [
    (vk::AccessFlags::INDIRECT_COMMAND_READ, Needs::Compute),
    (vk::AccessFlags::INDEX_READ, Needs::Graphics),
    (vk::AccessFlags::VERTEX_ATTRIBUTE_READ, Needs::Graphics),
    (vk::AccessFlags::UNIFORM_READ, Needs::Compute),
    (vk::AccessFlags::INPUT_ATTACHMENT_READ, Needs::Graphics),
    (vk::AccessFlags::SHADER_READ, Needs::Compute),
    (vk::AccessFlags::SHADER_WRITE, Needs::Compute),
    (vk::AccessFlags::COLOR_ATTACHMENT_READ, Needs::Graphics),
    (vk::AccessFlags::COLOR_ATTACHMENT_WRITE, Needs::Graphics),
    (
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
        Needs::Graphics,
    ),
    (
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        Needs::Graphics,
    ),
    (vk::AccessFlags::TRANSFER_READ, Needs::Transfer),
    (vk::AccessFlags::TRANSFER_WRITE, Needs::Transfer),
    (vk::AccessFlags::MEMORY_READ, Needs::Any),
    (vk::AccessFlags::MEMORY_WRITE, Needs::Any),
    (vk::AccessFlags::HOST_READ, Needs::Host),
    (vk::AccessFlags::HOST_WRITE, Needs::Host),
];

/// Every read a command may make of an image.
const IMAGE_READS: [vk::AccessFlags; 5] = [
    vk::AccessFlags::SHADER_READ,
    vk::AccessFlags::INPUT_ATTACHMENT_READ,
    vk::AccessFlags::COLOR_ATTACHMENT_READ,
    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
    vk::AccessFlags::TRANSFER_READ,
];

/// The accesses a barrier recorded for `family` may name after `ALL_COMMANDS`.
fn supported(family: vk::QueueFlags) -> vk::AccessFlags {
    let graphics = family.contains(vk::QueueFlags::GRAPHICS);
    let compute = graphics || family.contains(vk::QueueFlags::COMPUTE);
    let transfer = compute || family.contains(vk::QueueFlags::TRANSFER);
    let mut supported = vk::AccessFlags::empty();
    for (access, needs) in ACCESS_STAGES {
        let held = match needs {
            Needs::Any => true,
            Needs::Transfer => transfer,
            Needs::Compute => compute,
            Needs::Graphics => graphics,
            Needs::Host => false,
        };
        if held {
            supported |= access;
        }
    }
    supported
}

/// `barrier`, recorded for `family`, names no host or catch-all read and no access the family's
/// stages lack.
fn holds_for(what: &str, family: vk::QueueFlags, barrier: &vk::ImageMemoryBarrier<'_>) {
    let dst = barrier.dst_access_mask;
    assert!(
        !dst.intersects(HOST_OR_ANY_READ),
        "{what} on {:#x}: destination {:#x}",
        family.as_raw(),
        dst.as_raw()
    );
    let beyond = dst & !supported(family);
    assert!(
        beyond.is_empty(),
        "{what} on {:#x}: destination {:#x} names {:#x}, which the family's stages lack",
        family.as_raw(),
        dst.as_raw(),
        beyond.as_raw()
    );
}

/// Each destination an image barrier can take, through the one constructor every image barrier is
/// made by, on each kind of queue family, and the barriers built from it: a first move (plain, a
/// depth image's, and an acquire), and a caller's image's transitions into `GENERAL` and back
/// home.
#[test]
fn every_image_barrier_names_the_recording_familys_device_accesses() {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let layouts = (vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
    let write = vk::AccessFlags::MEMORY_WRITE;
    for family in FAMILIES {
        for dst in [
            ImageDst::Every,
            ImageDst::TransferRead,
            ImageDst::TransferWrite,
        ] {
            // A new destination fails to compile here until it is listed above.
            match dst {
                ImageDst::Every | ImageDst::TransferRead | ImageDst::TransferWrite => {}
            }
            let barrier = image_barrier(vk::Image::null(), range, layouts, write, dst, family);
            let made = barrier.dst_access_mask.as_raw();
            assert_eq!(made, dst.access(family).as_raw(), "{dst:?}");
            holds_for(&format!("{dst:?}"), family, &barrier);
        }
        // Every names each read of an image the family's commands make, and their writes.
        let every = ImageDst::Every.access(family);
        for read in IMAGE_READS
            .into_iter()
            .filter(|&read| supported(family).contains(read))
        {
            assert!(
                every.contains(read),
                "{:#x}: {:#x} lacks {:#x}",
                family.as_raw(),
                every.as_raw(),
                read.as_raw()
            );
        }
        assert!(every.contains(vk::AccessFlags::MEMORY_WRITE));

        let moved = FirstMove::new(
            vk::Image::null(),
            Format::R8G8B8A8_UNORM,
            vk::ImageLayout::UNDEFINED,
            None,
        );
        holds_for("a first move", family, &moved.barrier(family));
        let depth = FirstMove::new(
            vk::Image::null(),
            Format::D32_SFLOAT,
            vk::ImageLayout::PREINITIALIZED,
            Some(std::sync::Weak::<()>::new()),
        );
        holds_for("a depth image's first move", family, &depth.barrier(family));
        let acquired = FirstMove::new(
            vk::Image::null(),
            Format::R8G8B8A8_UNORM,
            vk::ImageLayout::UNDEFINED,
            None,
        )
        .acquired(0);
        holds_for("an acquire", family, &acquired.barrier(family));
        for (old, new) in [
            (ImageLayout::UNDEFINED, ImageLayout::GENERAL),
            (ImageLayout::GENERAL, ImageLayout::PRESENT_SRC_KHR),
            (ImageLayout::SHADER_READ_ONLY_OPTIMAL, ImageLayout::GENERAL),
        ] {
            let transition = Transition {
                image: vk::Image::null(),
                aspects: ImageAspectFlags::COLOR,
                old,
                new,
            };
            let barrier = transition.barrier(family);
            holds_for(&format!("{old:?} to {new:?}"), family, &barrier);
            let made = barrier.dst_access_mask.as_raw();
            assert_eq!(made, every.as_raw(), "{old:?} to {new:?}");
        }
    }
}

/// [`ImageDst::Every`] by kind of family: a transfer's read on a family of transfers alone, a
/// shader's too on a compute family, and the attachments' too on a graphics family; each with
/// the writes.
#[test]
fn every_follows_the_family_kind() {
    let reads = |reads: &[vk::AccessFlags]| {
        (reads.iter()).fold(vk::AccessFlags::MEMORY_WRITE, |mask, read| mask | *read)
    };
    let transfer = reads(&[vk::AccessFlags::TRANSFER_READ]);
    let compute = reads(&[vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_READ]);
    let graphics = reads(&IMAGE_READS);
    for (family, expected) in FAMILIES
        .into_iter()
        .zip([transfer, transfer, compute, compute, graphics, graphics])
    {
        let every = ImageDst::Every.access(family).as_raw();
        assert_eq!(every, expected.as_raw(), "{:#x}", family.as_raw());
    }
}
