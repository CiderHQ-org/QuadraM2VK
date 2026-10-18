//! What each texture creator answers for the descriptors whose answers differ among Metal's
//! creators (the consumer's sweep of 14 creators x 73 formats), on the device the cfg selects,
//! under the validation layer. Metal's answers are plain Metal's on an Apple M4 under macOS 27.0.1
//! and an M3 Max under 26.7, alike:
//!
//! - `newSharedTextureWithDescriptor:` answers nil for storage other than `Private` before any
//!   rule of the descriptor (a width of 0: nil in `Shared` storage, a stop in `Private`).
//! - A texture buffer is at most 2^28 texels wide: one more is stopped by every creator and the
//!   heap size query, which answers 2^28 bytes at 16 for an `R8Unorm` one of 2^28.
//! - A multisample texture of one sample, or of a count neither the device nor an Apple GPU
//!   takes (3; 8 and 16 where the device does not take them), is stopped; an `RGBA16Float_XR`
//!   one at such a count is nil from the shared creator and stopped by the device's.
//! - A texture buffer of 2 levels is made by Metal's device, heap and shared creators, which the
//!   runtime refuses as its gap (`NotImplemented(TextureBufferLevels)`); a buffer's creator
//!   stops it, and the heap size query answers its first level's size (256 bytes at 16 for 64
//!   `RGBA8Unorm` texels).
//! - A swizzle with writes under `writeSwizzleEnabled` is made by Metal's device, heap and
//!   buffer creators (the runtime's gap, `NotImplemented(WriteSwizzle)`) and stopped by the
//!   shared creator; the heap size query answers. Without the property every creator stops it.

use super::{validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};
use metal2vulkan_runtime::{
    BufferTextureProblem, Device, Error, Heap, HeapDescriptor, HeapType, Invalid, Nil, Problem,
    ResourceOptions, RuntimeGap, TextureDescriptor, TextureProblem,
};

fn invalid(call: &'static str, problem: TextureProblem) -> Error {
    Error::Invalid(Invalid::new(call, Problem::Texture(problem)))
}

/// The answers of the device's, a private automatic heap's and the shared creator, and of the
/// heap size query, for `descriptor` (`Private`).
fn answers(device: &Device, heap: &Heap, descriptor: &TextureDescriptor) -> [Option<Error>; 4] {
    [
        device.new_texture(descriptor).err(),
        heap.new_texture(descriptor).err(),
        device.new_shared_texture(descriptor).err(),
        device.heap_texture_size_and_align(descriptor).err(),
    ]
}

fn private(mut descriptor: TextureDescriptor) -> TextureDescriptor {
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

#[test]
fn each_creator_answers_as_metals() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 20,
                ResourceOptions::private(),
            ))
            .unwrap();
        let buffer = device
            .new_buffer(1 << 16, ResourceOptions::shared())
            .unwrap();

        // The shared creator: nil for storage other than `Private` first.
        let mut zero = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
        zero.width = 0;
        assert_eq!(
            device.new_shared_texture(&zero).err(),
            Some(Error::Nil(Nil::new(
                "new_shared_texture",
                Problem::StorageMode
            )))
        );
        assert_eq!(
            device.new_shared_texture(&private(zero)).err(),
            Some(invalid("new_shared_texture", TextureProblem::ZeroExtent))
        );

        // A texture buffer's width.
        let wide = |width| {
            private(TextureDescriptor::texture_buffer(
                PixelFormat::R8Unorm,
                width,
                ResourceOptions::private(),
                TextureUsage::SHADER_READ,
            ))
        };
        let past = TextureProblem::PastLargestSide { max: 1 << 28 };
        let [made, heaped, shared, sized] = answers(&device, &heap, &wide((1 << 28) + 1));
        assert_eq!(made, Some(invalid("new_texture", past)));
        assert_eq!(heaped, Some(invalid("new_texture", past)));
        assert_eq!(shared, Some(invalid("new_shared_texture", past)));
        assert_eq!(sized, Some(invalid("heap_texture_size_and_align", past)));
        let widest = device.heap_texture_size_and_align(&wide(1 << 28)).unwrap();
        assert_eq!((widest.size, widest.align), (1 << 28, 16));

        // A multisample texture's sample count.
        let ms = |pf, samples| {
            let mut d = TextureDescriptor::texture_2d(pf, 16, 16, false);
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = samples;
            d.usage = TextureUsage::RENDER_TARGET;
            private(d)
        };
        let takes = |samples: u64| {
            device
                .properties()
                .supports_texture_sample_count(samples)
                .unwrap()
        };
        let counts = [1, 3, 8, 16].into_iter();
        for samples in counts.filter(|&n| n == 1 || !takes(n)) {
            let stop = TextureProblem::SampleCount;
            let [made, heaped, shared, sized] =
                answers(&device, &heap, &ms(PixelFormat::RGBA8Unorm, samples));
            assert_eq!(made, Some(invalid("new_texture", stop)), "{samples}");
            assert_eq!(heaped, Some(invalid("new_texture", stop)), "{samples}");
            assert_eq!(
                shared,
                Some(invalid("new_shared_texture", stop)),
                "{samples}"
            );
            assert_eq!(
                sized,
                Some(invalid("heap_texture_size_and_align", stop)),
                "{samples}"
            );
            let xr = ms(PixelFormat::RGBA16FloatXR, samples);
            assert_eq!(
                device.new_texture(&xr).err(),
                Some(invalid("new_texture", stop))
            );
            assert_eq!(
                device.new_shared_texture(&xr).err(),
                Some(Error::Nil(Nil::new(
                    "new_shared_texture",
                    Problem::Texture(TextureProblem::NoTextureOfFormat)
                ))),
                "{samples}"
            );
        }
        // An Apple GPU's count is never stopped: made, or the device's gap.
        for samples in [2, 4] {
            let made = device.new_texture(&ms(PixelFormat::RGBA8Unorm, samples));
            match made {
                Ok(_) => assert!(takes(samples)),
                Err(Error::Unsupported(_)) => assert!(!takes(samples)),
                Err(e) => panic!("{samples} samples: {e}"),
            }
        }

        // A texture buffer of 2 levels.
        let mut levels = TextureDescriptor::texture_buffer(
            PixelFormat::RGBA8Unorm,
            64,
            ResourceOptions::private(),
            TextureUsage::SHADER_READ,
        );
        levels.mipmap_level_count = 2;
        let gap = Some(Error::NotImplemented(RuntimeGap::TextureBufferLevels));
        let [made, heaped, shared, sized] = answers(&device, &heap, &levels);
        assert_eq!((made, heaped, shared), (gap.clone(), gap.clone(), gap));
        assert_eq!(sized, None);
        let answer = device.heap_texture_size_and_align(&levels).unwrap();
        assert_eq!((answer.size, answer.align), (256, 16));
        assert_eq!(
            buffer.new_texture(&levels, 0, 256).err(),
            Some(Error::Invalid(Invalid::new(
                "new_texture",
                Problem::BufferTexture(BufferTextureProblem::MipmapLevelCount { count: 2 })
            )))
        );

        // A swizzle with writes, under `writeSwizzleEnabled` and without it.
        let bgra = TextureSwizzleChannels::new(
            TextureSwizzle::Blue,
            TextureSwizzle::Green,
            TextureSwizzle::Red,
            TextureSwizzle::Alpha,
        );
        let mut swizzled = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
        swizzled.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        swizzled.swizzle = bgra;
        swizzled.write_swizzle_enabled = true;
        let swizzled = private(swizzled);
        let gap = Some(Error::NotImplemented(RuntimeGap::WriteSwizzle));
        let stop = TextureProblem::SwizzleWithWrites;
        let [made, heaped, shared, sized] = answers(&device, &heap, &swizzled);
        assert_eq!((made, heaped), (gap.clone(), gap.clone()));
        assert_eq!(shared, Some(invalid("new_shared_texture", stop)));
        assert_eq!(sized, None);
        let mut over = swizzled;
        over.resource_options = ResourceOptions::shared();
        assert_eq!(buffer.new_texture(&over, 0, 64).err(), gap);
        let mut off = swizzled;
        off.write_swizzle_enabled = false;
        let [made, heaped, shared, sized] = answers(&device, &heap, &off);
        assert_eq!(made, Some(invalid("new_texture", stop)));
        assert_eq!(heaped, Some(invalid("new_texture", stop)));
        assert_eq!(shared, Some(invalid("new_shared_texture", stop)));
        assert_eq!(sized, Some(invalid("heap_texture_size_and_align", stop)));
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}
