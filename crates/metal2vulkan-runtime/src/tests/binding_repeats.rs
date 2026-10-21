//! Repeated bindings keep their stamp without skipping validation or changes of slot kind.
use super::blit_holds::{device, Scratch};
use crate::argument_table::{ArgumentTable, ATTRIBUTE_STRIDE_STATIC};
use crate::encoder::EncoderCore;
use crate::{ResourceOptions, TextureDescriptor};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::PixelFormat;

#[test]
fn unchanged_buffers_textures_and_samplers_preserve_the_stamp() {
    let scratch = Scratch::new("binding-repeats");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let command = queue.command_buffer().unwrap();
    let mut core = EncoderCore::open(&command, "test", true).unwrap();
    let mut table = ArgumentTable::default();
    let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
    let texture = device
        .new_texture(&TextureDescriptor::texture_2d(
            PixelFormat::RGBA8Unorm,
            4,
            4,
            false,
        ))
        .unwrap();
    table
        .set_buffer(&mut core, "test", Some(&buffer), 16, 0)
        .unwrap();
    table
        .set_textures(&mut core, "test", &[Some(&texture)], 2)
        .unwrap();
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    table
        .set_samplers(&mut core, "test", &[Some(&sampler)], None, 0)
        .unwrap();
    let same = table.stamp();
    for _ in 0..200 {
        table
            .set_buffer(&mut core, "test", Some(&buffer), 16, 0)
            .unwrap();
        table
            .set_buffers(&mut core, "test", &[Some(&buffer)], &[16], 0)
            .unwrap();
        table
            .set_textures(&mut core, "test", &[None, Some(&texture)], 1)
            .unwrap();
        table
            .set_samplers(&mut core, "test", &[Some(&sampler)], None, 0)
            .unwrap();
        assert_eq!(table.stamp(), same);
    }
    assert!(table
        .set_buffer(&mut core, "test", Some(&buffer), 257, 0)
        .is_err());
    assert!(table.set_textures(&mut core, "test", &[None], 128).is_err());
    assert_eq!(table.stamp(), same, "a refused binding changes nothing");
    table
        .set_samplers(
            &mut core,
            "test",
            &[Some(&sampler)],
            Some((&[0.0], &[1.0])),
            0,
        )
        .unwrap();
    let clamped = table.stamp();
    assert_ne!(clamped, same);
    table
        .set_samplers(
            &mut core,
            "test",
            &[Some(&sampler)],
            Some((&[0.0], &[1.0])),
            0,
        )
        .unwrap();
    assert_eq!(table.stamp(), clamped);
    table
        .set_samplers(
            &mut core,
            "test",
            &[Some(&sampler)],
            Some((&[-0.0], &[1.0])),
            0,
        )
        .unwrap();
    assert_ne!(
        table.stamp(),
        clamped,
        "the sign of a LOD clamp's zero stays distinct"
    );
    let before_stride = table.stamp();
    table
        .set_buffer_with_stride(&mut core, "test", Some(&buffer), 16, 32, 0)
        .unwrap();
    assert_ne!(
        table.stamp(),
        before_stride,
        "dynamic stride changes the binding"
    );
    table
        .set_buffer_with_stride(
            &mut core,
            "test",
            Some(&buffer),
            16,
            ATTRIBUTE_STRIDE_STATIC,
            0,
        )
        .unwrap();
    table
        .set_bytes(&mut core, "test", &[1; 16], None, 0)
        .unwrap();
    let inline = table.stamp();
    table
        .set_buffer(&mut core, "test", Some(&buffer), 16, 0)
        .unwrap();
    assert_ne!(table.stamp(), inline, "a buffer must replace inline bytes");
    let bound = table.stamp();
    table
        .set_buffer(&mut core, "test", Some(&buffer), 32, 0)
        .unwrap();
    assert_ne!(table.stamp(), bound, "an offset changes the binding");
}
