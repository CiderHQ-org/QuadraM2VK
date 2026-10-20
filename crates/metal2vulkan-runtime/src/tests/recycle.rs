//! Recycled images (`texture/recycle.rs`), on the device the GPU cfg selects.
//!
//! - **A new texture reads what a new one reads.** A `Private` `BGRA8Unorm` render target, cleared
//!   to a colour and released, then a texture of the same descriptor: it takes the parked image
//!   (`WorkCounts::object_churn.images_recycled`), and a blit of it reads the same bytes as a
//!   texture of the same descriptor made while no image was parked, never the colour. Native
//!   Metal on the M3 Max under macOS 27.0.1 reads a fixed pattern from such a texture made right
//!   after one of the same shape was cleared and released, and zero from an uncompressed
//!   `Private` or a `Shared` one, never the old texels (the inplace-surface probe's `fresh` mode).
//! - **Each frame's textures, recycled.** Eight such textures made, cleared and released each of
//!   sixty frames take parked images from the second frame on: no image made after the first
//!   frame's, and none parked past the bounds.

use super::blit_holds::{Scratch, WAIT};
use crate::{
    BlitOption, ClearColor, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions,
    LoadAction, Origin, RenderPassDescriptor, ResourceOptions, ShaderCacheOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;

const SIDE: u64 = 64;

/// The device the cfg selects, its shader cache in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

fn descriptor() -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// Clear `target` to a colour, waited for.
fn clear(queue: &CommandQueue, target: &Texture) {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The bytes of `texture`, blitted out.
fn read(device: &Device, queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let length = SIDE * SIDE * 4;
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &out,
        0,
        SIDE * 4,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    // SAFETY: the shared buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), length as usize) }
        .to_vec()
}

/// Wait until the device parked `count` images in all.
fn parked_at_least(device: &Device, count: u64) {
    let until = std::time::Instant::now() + WAIT;
    while device.work_counts().object_churn.images_parked < count {
        assert!(std::time::Instant::now() < until, "the images were parked");
        std::thread::yield_now();
    }
}

#[test]
fn a_recycled_texture_reads_what_a_new_one_reads() {
    let scratch = Scratch::new("recycle-fresh");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    // A texture made while nothing is parked: what a new texture of the descriptor reads.
    let first = device.new_texture(&descriptor()).unwrap();
    let fresh = read(&device, &queue, &first);
    clear(&queue, &first);
    let cleared = read(&device, &queue, &first);
    assert_ne!(fresh, cleared, "the clear changed the texels");
    let before = device.work_counts().object_churn;
    drop(first);
    parked_at_least(&device, before.images_parked + 1);
    let second = device.new_texture(&descriptor()).unwrap();
    let after = device.work_counts().object_churn;
    assert_eq!(
        after.images_recycled - before.images_recycled,
        1,
        "the second texture took the first one's parked image"
    );
    assert_eq!(
        read(&device, &queue, &second),
        fresh,
        "the recycled texture reads what a new one reads, not the old texels"
    );
}

#[test]
fn each_frames_textures_take_parked_images() {
    let scratch = Scratch::new("recycle-frames");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    const FRAMES: u64 = 60;
    const TEXTURES: u64 = 8;
    let start = device.work_counts().object_churn;
    for frame in 1..=FRAMES {
        let textures: Vec<Texture> = (0..TEXTURES)
            .map(|_| device.new_texture(&descriptor()).unwrap())
            .collect();
        for texture in &textures {
            clear(&queue, texture);
        }
        drop(textures);
        // The images go as their command buffers' holds do, which a completion may let go of
        // after the wait returned.
        parked_at_least(&device, start.images_parked + frame * TEXTURES);
    }
    let end = device.work_counts().object_churn;
    let recycled = end.images_recycled - start.images_recycled;
    eprintln!(
        "RECYCLE: {FRAMES} frames of {TEXTURES} textures: {recycled} images recycled, {} parked, \
         {} evicted",
        end.images_parked - start.images_parked,
        end.images_evicted - start.images_evicted
    );
    assert_eq!(
        recycled,
        (FRAMES - 1) * TEXTURES,
        "every frame after the first recycles"
    );
}
