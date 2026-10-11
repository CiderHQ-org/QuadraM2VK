//! A placement heap with textures at offsets the program chooses: the device's size and alignment
//! for the texture, a `Private` heap of two, a texture at each offset, and fences that order the
//! encoders that use them (a heap's resources are untracked by default, as on Metal).
//!
//! ```sh
//! cargo run -p metal2vulkan-runtime --example heap_placement [-- path/to/library.metallib]
//! ```
//!
//! A blit uploads texels into the first texture, the kernel `invert_texels` of the repository's
//! owned fixture (`validation/fixtures/public/examples/runtime_examples.metal`, or of the metallib
//! the argument names) writes their inverse into the second, and a blit reads it back.

mod common;

use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Device, HeapDescriptor, HeapType, Origin, PipelineOptions, ResourceOptions, Size,
    TextureDescriptor,
};

/// The textures' side, in texels.
const SIDE: u64 = 16;

fn main() -> common::Result<()> {
    let device = Device::system_default()?;
    println!("device: {}", device.name());
    let library = common::library(&device)?;
    let function = library.new_function_with_name("invert_texels")?;
    let state =
        device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?;

    // A texture's storage and CPU cache modes are its heap's.
    let mut texture = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    texture.usage = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);
    texture.resource_options = ResourceOptions::private();

    // `heapTextureSizeAndAlignWithDescriptor:`: this device's own requirement for the texture,
    // which is not an Apple GPU's. A program places by the answer, never by a constant.
    let need = device.heap_texture_size_and_align(&texture)?;
    let stride = need.size.next_multiple_of(need.align);
    println!(
        "a {SIDE}x{SIDE} RGBA8Unorm texture takes {} bytes at a multiple of {}",
        need.size, need.align
    );

    // `MTLHeapTypePlacement`: the program places each resource at an offset.
    let heap = device.new_heap(&HeapDescriptor::new(
        HeapType::Placement,
        2 * stride,
        ResourceOptions::private(),
    ))?;
    // `newTextureWithDescriptor:offset:`.
    let source = heap.new_texture_at_offset(&texture, 0)?;
    let destination = heap.new_texture_at_offset(&texture, stride)?;
    println!(
        "heap of {} bytes, hazard tracking {:?}; textures at 0 and {stride}",
        heap.size(),
        heap.hazard_tracking_mode()
    );

    let row = SIDE * 4;
    let texels: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| [i as u8, (i * 3) as u8, 255 - i as u8, 200])
        .collect();
    let upload = device.new_buffer_with_bytes(&texels, ResourceOptions::shared())?;
    let download = device.new_buffer(row * SIDE, ResourceOptions::shared())?;

    // The heap's resources are untracked: the encoders order themselves with fences
    // (`updateFence:` after the writes, `waitForFence:` before the reads), as on Metal.
    let uploaded = device.new_fence();
    let inverted = device.new_fence();
    let queue = device.new_command_queue()?;
    let command_buffer = queue.command_buffer()?;

    let mut blit = command_buffer.blit_command_encoder()?;
    blit.copy_from_buffer_to_texture(
        &upload,
        0,
        row,
        row * SIDE,
        Size::new(SIDE, SIDE, 1),
        &source,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )?;
    blit.update_fence(&uploaded)?;
    blit.end_encoding()?;

    let mut encoder = command_buffer.compute_command_encoder()?;
    encoder.wait_for_fence(&uploaded)?;
    encoder.set_compute_pipeline_state(&state)?;
    encoder.set_texture(Some(&source), 0)?;
    encoder.set_texture(Some(&destination), 1)?;
    encoder.dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))?;
    encoder.update_fence(&inverted)?;
    encoder.end_encoding()?;

    let mut blit = command_buffer.blit_command_encoder()?;
    blit.wait_for_fence(&inverted)?;
    blit.copy_from_texture_to_buffer(
        &destination,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &download,
        0,
        row,
        row * SIDE,
        BlitOption::None,
    )?;
    blit.end_encoding()?;
    common::run(&command_buffer)?;

    // Each color channel inverted, alpha kept.
    let read = common::bytes(&download)?;
    for (at, (texel, original)) in read
        .as_chunks::<4>()
        .0
        .iter()
        .zip(texels.as_chunks::<4>().0.iter())
        .enumerate()
    {
        let expected = [
            255 - original[0],
            255 - original[1],
            255 - original[2],
            original[3],
        ];
        if *texel != expected {
            return Err(format!("texel {at} is {texel:?}, not {expected:?}").into());
        }
    }
    println!(
        "texel 1 was {:?} and is {:?}: every texel as expected",
        &texels[4..8],
        &read[4..8]
    );
    Ok(())
}
