//! Device tests of the runtime's textures on the device the cfg selects: textures of every pixel
//! format made or refused typed, CPU copies through `replace_region` and `get_bytes`, views of
//! levels, slices and formats, a view of another component layout read in a kernel, and Metal's
//! refusals, under the validation layer.

use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::candidate::runtime_samples::encode_through;
use crate::candidate::runtime_samples::SampleFill;
use crate::case::{TextureFormat as CaseFormat, TextureType as CaseType};
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::gpu::TEST_DEVICE;
use crate::literal::texture_layout;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBufferStatus, CopyProblem, Device, Error, HeapDescriptor, HeapType,
    Invalid, Origin, Problem, Region, ResourceOptions, Size, StorageMode, Texture,
    TextureDescriptor, TextureProblem, ViewProblem,
};

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// `n` bytes of a pattern seeded by `seed`.
pub(super) fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// The bytes of one texel of `pf`, for the uncompressed color formats this test round-trips:
/// none for a format whose image does not hold Metal's bytes (`byte_format`), whose CPU copies
/// the runtime refuses.
pub(super) fn texel_bytes(pf: PixelFormat) -> Option<usize> {
    let block = metal2vulkan::metal::format::byte_format(pf)?.block()?;
    let compressed =
        block.texels > 1 || (block.packed_bits.is_none() && block.component_bits.is_none());
    (block.color.is_some() && !compressed).then_some(block.size as usize)
}

/// Whether Metal makes no texture of `pf` from a descriptor alone, answering nil to the device's
/// and a heap's calls (`TextureProblem::NoTextureOfFormat`), where measured on an Apple M4 under
/// macOS 27.0.1: the two-plane YCbCr formats (each of the 69), made only over an IOSurface's
/// planes, and `RGBA16Float_XR`, `AnyDepth` and `AnyStencil`.
pub(super) fn no_texture_alone(pf: PixelFormat) -> bool {
    metal2vulkan::metal::format::ycbcr_format(pf).is_some()
        || matches!(
            pf,
            PixelFormat::RGBA16FloatXR | PixelFormat::AnyDepth | PixelFormat::AnyStencil
        )
}

/// `R10Unorm_X6` and `RG10Unorm_X12`, and their `_sRGB` twins, are made only over memory a texture
/// is given, as Metal's textures of them store nothing in memory of their own: every creator of a
/// texture in memory of its own refuses them `NotImplemented(Format)`
/// naming the format (the device's `new_texture`, an automatic heap's `new_texture`,
/// `new_shared_texture`, each for a 2D texture and a texture buffer, and the heap size query), and
/// a caller's buffer's `new_texture` makes them.
#[test]
fn padded_formats_are_made_only_over_given_memory() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let private = ResourceOptions::private();
        let heap = device
            .new_heap(&HeapDescriptor::new(HeapType::Automatic, 1 << 20, private))
            .unwrap();
        for pf in [
            PixelFormat::R10UnormX6,
            PixelFormat::RG10UnormX12,
            PixelFormat::R10UnormX6SRGB,
            PixelFormat::RG10UnormX12SRGB,
        ] {
            let refused = Some(Error::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(pf),
            ));
            let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let mut plain = TextureDescriptor::texture_2d(pf, 16, 16, false);
            plain.usage = usage;
            plain.resource_options = private;
            let texels = TextureDescriptor::texture_buffer(pf, 64, private, usage);
            for (what, d) in [("2D", &plain), ("texture buffer", &texels)] {
                let what = format!("{pf:?} {what}");
                assert_eq!(device.new_texture(d).err(), refused, "{what}: the device's");
                assert_eq!(heap.new_texture(d).err(), refused, "{what}: the heap's");
                assert_eq!(
                    device.new_shared_texture(d).err(),
                    refused,
                    "{what}: a shareable one"
                );
                assert_eq!(
                    device.heap_texture_size_and_align(d).err(),
                    refused,
                    "{what}: the heap size query"
                );
            }
            let bpr = 256;
            let buffer = device
                .new_buffer(bpr * 16, ResourceOptions::shared())
                .unwrap();
            let mut over = plain;
            over.resource_options = ResourceOptions::shared();
            let texture = buffer
                .new_texture(&over, 0, bpr)
                .unwrap_or_else(|e| panic!("{pf:?} over a buffer: {e}"));
            assert_eq!(texture.pixel_format(), pf);
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// A 2D texture of `pf` and `extent`, read by shaders, with `levels` levels.
fn texture_2d(
    device: &Device,
    pf: PixelFormat,
    extent: u64,
    levels: u64,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, extent, extent, false);
    d.mipmap_level_count = levels;
    device.new_texture(&d)
}

/// Every pixel format makes a texture or is refused typed, and a format the device's Metal level
/// describes none of is stopped (`NotAFormat`), the 24-bit depth formats on every venue; every
/// uncompressed color format's new `Shared` texture reads zero, as Metal's do, takes bytes through
/// `replace_region` and gives the same bytes back through `get_bytes`, with padded rows left alone.
#[test]
fn every_format_makes_a_texture_or_is_refused_typed() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (mut made, mut refused, mut round_trips) = (0, 0, 0);
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // The device's formats: its Metal level's (the 24-bit depth formats at none).
        let level = device.properties().metal_level;
        let describes = |pf: PixelFormat| level.describes(pf);
        for &pf in PixelFormat::ALL {
            match texture_2d(&device, pf, 8, 1) {
                Ok(texture) => {
                    made += 1;
                    assert!(describes(pf), "{pf:?} made");
                    assert_eq!(texture.pixel_format(), pf);
                    assert_eq!(
                        (texture.width(), texture.height(), texture.depth()),
                        (8, 8, 1)
                    );
                    let Some(bytes) = texel_bytes(pf) else {
                        continue;
                    };
                    // Rows of 8 texels, a texel of padding after each.
                    let row = 8 * bytes;
                    let bpr = row + bytes;
                    let region = Region::new_2d(0, 0, 8, 8);
                    let mut fresh = vec![0xeeu8; bpr * 8];
                    texture
                        .get_bytes(&mut fresh, bpr as u64, 0, region, 0, 0)
                        .unwrap_or_else(|e| panic!("{pf:?}: {e}"));
                    for y in 0..8 {
                        assert!(
                            fresh[y * bpr..y * bpr + row].iter().all(|&b| b == 0),
                            "{pf:?}: a new Shared texture reads zero"
                        );
                    }
                    let src = pattern(bpr * 8, pf.as_raw() as u8);
                    texture
                        .replace_region(region, 0, 0, &src, bpr as u64, 0)
                        .unwrap_or_else(|e| panic!("{pf:?}: {e}"));
                    let mut back = vec![0xeeu8; bpr * 8];
                    texture
                        .get_bytes(&mut back, bpr as u64, 0, region, 0, 0)
                        .unwrap_or_else(|e| panic!("{pf:?}: {e}"));
                    for y in 0..8 {
                        let r = y * bpr..y * bpr + row;
                        assert_eq!(back[r.clone()], src[r], "{pf:?} row {y}");
                        assert!(back[y * bpr + row..(y + 1) * bpr]
                            .iter()
                            .all(|&b| b == 0xee));
                    }
                    round_trips += 1;
                }
                // `Invalid`, and every format the device's Metal describes none of, which
                // Metal's validation stops as it stops `Invalid` ("has invalid pixelFormat").
                Err(Error::Invalid(Invalid {
                    problem: Problem::Texture(TextureProblem::NotAFormat),
                    ..
                })) => {
                    assert!(!describes(pf), "{pf:?} stopped");
                    refused += 1;
                }
                // Metal makes no such texture from a descriptor alone: it answers nil (a
                // two-plane YCbCr texture is made over a surface only).
                Err(Error::Nil(nil)) => {
                    let want = match pf {
                        PixelFormat::Unspecialized => TextureProblem::Unspecialized,
                        _ => {
                            assert!(no_texture_alone(pf), "{pf:?}");
                            TextureProblem::NoTextureOfFormat
                        }
                    };
                    assert_eq!(nil.problem, Problem::Texture(want), "{pf:?}");
                    refused += 1;
                }
                Err(Error::Unsupported(u)) => {
                    assert!(
                        matches!(u.feature, Feature::Format | Feature::TextureSize),
                        "{pf:?}: {u:?}"
                    );
                    refused += 1;
                }
                Err(other) => panic!("{pf:?}: {other}"),
            }
        }
        eprintln!(
            "textures on {}: {made} formats made, {refused} refused, {round_trips} round trips",
            device.name()
        );
        assert!(made > 50 && round_trips > 40, "{made} {round_trips}");
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// Levels, slices and 3D images copy where Metal puts them; views present their texture's levels
/// and slices and another format of the same bytes; Metal's refusals are typed.
#[test]
fn textures_copy_levels_slices_and_views() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let rgba = PixelFormat::RGBA8Unorm;

        // A 2D array of 3 slices and 3 levels: each level and slice its own bytes.
        let mut d = TextureDescriptor::texture_2d(rgba, 8, 8, true);
        d.texture_type = TextureType::Type2DArray;
        d.array_length = 3;
        d.mipmap_level_count = 3;
        d.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        let array = device.new_texture(&d).unwrap();
        assert_eq!((array.array_length(), array.mipmap_level_count()), (3, 3));
        assert_eq!(array.storage_mode(), StorageMode::Shared);
        for level in 0..3u64 {
            let side = 8 >> level;
            for slice in 0..3u64 {
                let src = pattern((side * side * 4) as usize, (level * 16 + slice) as u8);
                array
                    .replace_region(
                        Region::new_2d(0, 0, side, side),
                        level,
                        slice,
                        &src,
                        side * 4,
                        0,
                    )
                    .unwrap();
            }
        }
        let read = |t: &Texture, level: u64, slice: u64, side: u64| {
            let mut b = vec![0u8; (side * side * 4) as usize];
            t.get_bytes(
                &mut b,
                side * 4,
                0,
                Region::new_2d(0, 0, side, side),
                level,
                slice,
            )
            .unwrap();
            b
        };
        for level in 0..3u64 {
            for slice in 0..3u64 {
                let side = 8 >> level;
                let want = pattern((side * side * 4) as usize, (level * 16 + slice) as u8);
                assert_eq!(read(&array, level, slice, side), want, "{level}/{slice}");
            }
        }

        // A view of levels 1..3 and slices 1..3 as 2D array, then a 2D view of its level 1 and
        // slice 1 (the texture's level 2, slice 2), in another format of the same bytes.
        let view = array
            .new_texture_view_with_levels_slices(rgba, TextureType::Type2DArray, 1..3, 1..3)
            .unwrap();
        assert_eq!(
            (view.width(), view.mipmap_level_count(), view.array_length()),
            (4, 2, 2)
        );
        assert_eq!(
            (view.parent_relative_level(), view.parent_relative_slice()),
            (1, 1)
        );
        let inner = view
            .new_texture_view_with_levels_slices(
                PixelFormat::R32Uint,
                TextureType::Type2D,
                1..2,
                1..2,
            )
            .unwrap();
        assert_eq!(inner.parent_texture(), Some(&view));
        assert_eq!(
            (inner.width(), inner.pixel_format()),
            (2, PixelFormat::R32Uint)
        );
        assert_eq!(
            read(&inner, 0, 0, 2),
            pattern(16, 2 * 16 + 2),
            "the view's level 0 and slice 0 are the texture's level 2 and slice 2"
        );
        let srgb = array.new_texture_view(PixelFormat::RGBA8UnormSRGB).unwrap();
        assert_eq!(read(&srgb, 1, 2, 4), pattern(64, 16 + 2));

        // Writing through a view writes the texture.
        let src = pattern(16, 0x55);
        inner
            .replace_region(Region::new_2d(0, 0, 2, 2), 0, 0, &src, 8, 0)
            .unwrap();
        assert_eq!(read(&array, 2, 2, 2), src);

        // A 3D texture, with images apart by more than their rows.
        let mut d = TextureDescriptor::texture_2d(rgba, 4, 4, false);
        d.texture_type = TextureType::Type3D;
        d.depth = 3;
        let volume = device.new_texture(&d).unwrap();
        let (bpr, bpi) = (16u64, 80u64);
        let src = pattern((bpi * 3) as usize, 9);
        let region = Region::new_3d(0, 0, 0, 4, 4, 3);
        volume.replace_region(region, 0, 0, &src, bpr, bpi).unwrap();
        let mut back = vec![0u8; (bpi * 3) as usize];
        volume.get_bytes(&mut back, bpr, bpi, region, 0, 0).unwrap();
        for z in 0..3 {
            let image = (z * bpi) as usize..(z * bpi + 64) as usize;
            assert_eq!(back[image.clone()], src[image], "image {z}");
        }
        // A `bytesPerImage` off Metal's rules, as plain Metal takes it:
        // `get_bytes` writes images exactly that far apart, in turn, a later image's bytes over an
        // earlier one's; a blit reads and writes each row at its offset rounded down to 4 bytes,
        // out of the texture in turn too.
        let images: Vec<&[u8]> = (0..3).map(|z| &src[z * 80..z * 80 + 64]).collect();
        let in_turn = |bpi: usize, len: usize| {
            let mut want = vec![0xabu8; len];
            for (z, image) in images.iter().enumerate() {
                let at = z * bpi - z * bpi % 4;
                want[at..at + 64].copy_from_slice(image);
            }
            want
        };
        let mut close = vec![0xabu8; 136];
        volume.get_bytes(&mut close, bpr, 36, region, 0, 0).unwrap();
        assert_eq!(close, in_turn(36, 136), "get_bytes at 36");
        let queue = device.new_command_queue().unwrap();
        let size = Size::new(4, 4, 3);
        for bpi in [0u64, 36, 70] {
            let len = (2 * bpi + 64) as usize;
            let out = buffer_with(&device, &vec![0xab; len]).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut e = command_buffer.blit_command_encoder().unwrap();
            e.copy_from_texture_to_buffer(
                &volume,
                0,
                0,
                Origin::new(0, 0, 0),
                size,
                &out,
                0,
                bpr,
                bpi,
                BlitOption::None,
            )
            .unwrap();
            e.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let written = read_buffer(&out, 0, len as u64).unwrap();
            assert_eq!(written, in_turn(bpi as usize, len), "a blit out at {bpi}");
        }

        // A texture of more than half a memory block, placed in a block of its own.
        let big = texture_2d(&device, rgba, 4096, 1).unwrap();
        let src = pattern(4 * 4096, 3);
        let last_row = Region::new_2d(0, 4095, 4096, 1);
        big.replace_region(last_row, 0, 0, &src, 4 * 4096, 0)
            .unwrap();
        let mut back = vec![0u8; 4 * 4096];
        big.get_bytes(&mut back, 4 * 4096, 0, last_row, 0, 0)
            .unwrap();
        assert_eq!(back, src);
        drop(big);

        // Depth-only and stencil-only formats copy their texels (depth in [0, 1], which every
        // device keeps); a combined depth/stencil format is refused CPU access, as Metal does.
        for (pf, bytes) in [
            (PixelFormat::Depth16Unorm, pattern(8, 1)),
            (
                PixelFormat::Depth32Float,
                [0.0f32, 0.25, 0.5, 1.0]
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect(),
            ),
            (PixelFormat::Stencil8, pattern(4, 2)),
        ] {
            let texture = match texture_2d(&device, pf, 2, 1) {
                Ok(texture) => texture,
                Err(Error::Unsupported(u)) => {
                    eprintln!("{pf:?} refused: {u:?}");
                    continue;
                }
                Err(other) => panic!("{pf:?}: {other}"),
            };
            let bpr = bytes.len() as u64 / 2;
            let region = Region::new_2d(0, 0, 2, 2);
            let mut fresh = vec![0xeeu8; bytes.len()];
            texture.get_bytes(&mut fresh, bpr, 0, region, 0, 0).unwrap();
            assert!(fresh.iter().all(|&b| b == 0), "{pf:?} reads zero when new");
            texture
                .replace_region(region, 0, 0, &bytes, bpr, 0)
                .unwrap();
            let mut back = vec![0u8; bytes.len()];
            texture.get_bytes(&mut back, bpr, 0, region, 0, 0).unwrap();
            assert_eq!(back, bytes, "{pf:?}");
        }
        // A combined depth/stencil format copies its depth, 4 bytes a texel, at any pitch, the
        // stencil left as it is, as plain Metal does.
        if let Ok(combined) = texture_2d(&device, PixelFormat::Depth32FloatStencil8, 2, 1) {
            let queue = device.new_command_queue().unwrap();
            let blit = |buffer: &Buffer, into: bool, option: BlitOption, bpr: u64| {
                let command_buffer = queue.command_buffer().unwrap();
                let mut e = command_buffer.blit_command_encoder().unwrap();
                let (origin, size) = (Origin::new(0, 0, 0), Size::new(2, 2, 1));
                if into {
                    e.copy_from_buffer_to_texture(
                        buffer, 0, bpr, 0, size, &combined, 0, 0, origin, option,
                    )
                } else {
                    e.copy_from_texture_to_buffer(
                        &combined, 0, 0, origin, size, buffer, 0, bpr, 0, option,
                    )
                }
                .unwrap();
                e.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
            };
            let stencil = buffer_with(&device, &[1, 2, 3, 4]).unwrap();
            blit(&stencil, true, BlitOption::StencilFromDepthStencil, 2);
            let floats = |v: [f32; 2]| v.iter().flat_map(|f| f.to_ne_bytes()).collect::<Vec<u8>>();
            let depth = [
                floats([0.125, 0.375]),
                vec![0x99; 2],
                floats([0.625, 0.875]),
            ]
            .concat();
            let region = Region::new_2d(0, 0, 2, 2);
            combined
                .replace_region(region, 0, 0, &depth, 10, 0)
                .unwrap();
            let mut back = vec![0xabu8; 20];
            combined.get_bytes(&mut back, 10, 0, region, 0, 0).unwrap();
            let want = [&depth[..8], &[0xab; 2], &depth[10..], &[0xab; 2]].concat();
            assert_eq!(back, want, "depth through the CPU");
            let read = buffer_with(&device, &[0; 4]).unwrap();
            blit(&read, false, BlitOption::StencilFromDepthStencil, 2);
            assert_eq!(
                read_buffer(&read, 0, 4).unwrap(),
                [1, 2, 3, 4],
                "stencil kept"
            );
        }

        // Metal's refusals.
        let mut d = TextureDescriptor::texture_2d(rgba, 4, 4, false);
        d.resource_options = ResourceOptions::private();
        let private = device.new_texture(&d).unwrap();
        assert_eq!(private.storage_mode(), StorageMode::Private);
        assert_eq!(
            private.replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[0; 4], 4, 0),
            Err(invalid(
                "replace_region",
                Problem::Copy(CopyProblem::CpuAccessPrivate)
            ))
        );
        let small = texture_2d(&device, rgba, 4, 1).unwrap();
        assert_eq!(
            small.replace_region(Region::new_2d(2, 0, 4, 1), 0, 0, &[0; 16], 16, 0),
            Err(invalid(
                "replace_region",
                Problem::Copy(CopyProblem::Region)
            ))
        );
        // A pitch between 0 and a row copies nothing, as plain Metal copies nothing: the new
        // texture still reads zero.
        assert_eq!(
            small.replace_region(Region::new_2d(0, 0, 4, 2), 0, 0, &[0xab; 32], 8, 0),
            Ok(())
        );
        let mut kept = vec![0xeeu8; 64];
        small
            .get_bytes(&mut kept, 16, 0, Region::new_2d(0, 0, 4, 4), 0, 0)
            .unwrap();
        assert!(
            kept.iter().all(|&b| b == 0),
            "a short pitch wrote: {kept:?}"
        );
        // An empty region writes nothing, as plain Metal does.
        assert_eq!(
            small.replace_region(Region::new_2d(0, 0, 0, 2), 0, 0, &[0; 32], 16, 0),
            Ok(())
        );
        assert_eq!(
            small.replace_region(Region::new_2d(0, 0, 4, 2), 0, 0, &[0; 20], 16, 0),
            Err(invalid(
                "replace_region",
                Problem::Copy(CopyProblem::BufferRange)
            ))
        );
        assert_eq!(
            device
                .new_texture(&TextureDescriptor::texture_2d(rgba, 0, 4, false))
                .err(),
            Some(invalid(
                "new_texture",
                Problem::Texture(TextureProblem::ZeroExtent)
            ))
        );
        // A view of another format of the class without `PixelFormatView` usage, which plain
        // Metal makes (its validation stops it): the CPU reads the texture's bytes through it.
        let words = pattern(16, 9);
        small
            .replace_region(Region::new_2d(0, 0, 4, 1), 0, 0, &words, 16, 0)
            .unwrap();
        let view = small.new_texture_view(PixelFormat::R32Uint).unwrap();
        assert_eq!(view.pixel_format(), PixelFormat::R32Uint);
        let mut through = vec![0u8; 16];
        view.get_bytes(&mut through, 16, 0, Region::new_2d(0, 0, 4, 1), 0, 0)
            .unwrap();
        assert_eq!(through, words);
        assert_eq!(
            array
                .new_texture_view_with_levels_slices(rgba, TextureType::Type2D, 0..1, 0..2)
                .err(),
            Some(invalid(
                "new_texture_view_with_levels_slices",
                Problem::TextureView(ViewProblem::SliceCount)
            ))
        );
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// A view of another component layout presents its texture's bits, as Metal's does: a 4x
/// `RGBA8Unorm` texture filled sample by sample, viewed as `R32Uint` and read sample by sample in
/// a kernel, gives each texel's four bytes as one word. A device whose Vulkan views reinterpret no
/// image's components (a portability subset without `imageViewFormatReinterpretation`) refuses
/// the view, typed; MoltenVK's subset has it, and the runtime enables it.
#[test]
fn a_view_of_another_component_layout_reads_the_textures_bits() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (extent, samples) = ([3u32, 2u32], 4u32);
    // Each pixel's samples in order, rows in order: four bytes a texel.
    let bytes = pattern((extent[0] * extent[1] * samples * 4) as usize, 7);
    let answer;
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let mut d = TextureDescriptor::texture_2d(
            PixelFormat::RGBA8Unorm,
            extent[0].into(),
            extent[1].into(),
            false,
        );
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = samples.into();
        d.usage = TextureUsage::RENDER_TARGET
            | TextureUsage::SHADER_READ
            | TextureUsage::PIXEL_FORMAT_VIEW;
        d.resource_options = ResourceOptions::private();
        let texture = device.new_texture(&d).unwrap();
        answer = match texture.new_texture_view(PixelFormat::R32Uint) {
            Err(Error::Unsupported(u)) if u.feature == Feature::ViewFormatReinterpretation => {
                "refused, typed".to_owned()
            }
            Err(error) => panic!("new_texture_view: {error}"),
            Ok(view) => {
                let layout =
                    texture_layout(CaseType::D2Multisample, [extent[0], extent[1], 1], samples)
                        .unwrap();
                let fill =
                    SampleFill::new(&device, &texture, CaseFormat::Rgba8Unorm, layout, &bytes)
                        .unwrap();
                let queue = device.new_command_queue().unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                fill.encode(&command_buffer).unwrap();
                let readback = encode_through(
                    &device,
                    &command_buffer,
                    &view,
                    "rt6_read_words1",
                    SampleRead::Color(ReadbackView::R32Uint),
                    extent,
                    samples,
                )
                .unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                assert_eq!(
                    readback.bytes().unwrap(),
                    bytes,
                    "the R32Uint view's words are not the texture's bytes"
                );
                "the texture's bits".to_owned()
            }
        };
        eprintln!(
            "view reinterpretation on {}: RGBA8Unorm as R32Uint, {answer}",
            device.name()
        );
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// Labels and purgeable states of textures as Metal answered them (an Apple M3 Max and an M4): each
/// texture object's label its own, an opened shared texture's starting as its texture's; the state
/// kept by a texture of 16 KiB or more and by a shareable texture (shared with the texture opened
/// from its handle), `Empty` keeping the texels; `NonVolatile` to every request from a smaller
/// texture, a view, and a texture over a buffer, whose buffer keeps its own state, and from a
/// heap's texture, whose heap keeps its own.
#[test]
fn texture_labels_and_purgeable_states_answer_as_metal() {
    use metal2vulkan_runtime::PurgeableState::{Empty, KeepCurrent, NonVolatile, Volatile};
    use metal2vulkan_runtime::{HeapDescriptor, HeapType, IdPlacement, PurgeableState};
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let rgba = PixelFormat::RGBA8Unorm;
        let sequence = |texture: &Texture| -> Vec<PurgeableState> {
            [
                KeepCurrent,
                Volatile,
                KeepCurrent,
                Empty,
                KeepCurrent,
                NonVolatile,
                KeepCurrent,
            ]
            .into_iter()
            .map(|state| texture.set_purgeable_state(state).unwrap())
            .collect()
        };
        let kept = [
            NonVolatile,
            NonVolatile,
            Volatile,
            Volatile,
            Empty,
            Empty,
            NonVolatile,
        ];
        let never = [NonVolatile; 7];

        // A 64x64 texture (16 KiB) keeps its state, and `Empty` keeps its texels.
        let big = device
            .new_texture(&TextureDescriptor::texture_2d(rgba, 64, 64, false))
            .unwrap();
        assert_eq!(big.label(), None);
        big.set_label("big");
        assert_eq!(big.label().as_deref(), Some("big"));
        let texels = pattern(64 * 64 * 4, 7);
        let whole = Region::new_2d(0, 0, 64, 64);
        big.replace_region(whole, 0, 0, &texels, 256, 0).unwrap();
        assert_eq!(sequence(&big), kept);
        let mut read = vec![0; texels.len()];
        big.get_bytes(&mut read, 256, 0, whole, 0, 0).unwrap();
        assert_eq!(read, texels);

        // A 32x32 texture (4 KiB) answers `NonVolatile` to every request.
        let small = device
            .new_texture(&TextureDescriptor::texture_2d(rgba, 32, 32, false))
            .unwrap();
        assert_eq!(sequence(&small), never);

        // A view's label and state are its own: `NonVolatile` while its texture is volatile.
        let view = big.new_texture_view(rgba).unwrap();
        assert_eq!(view.label(), None);
        view.set_label("view");
        assert_eq!(big.label().as_deref(), Some("big"));
        assert_eq!(big.set_purgeable_state(Volatile), Ok(NonVolatile));
        assert_eq!(sequence(&view), never);
        assert_eq!(big.set_purgeable_state(NonVolatile), Ok(Volatile));

        // A texture over a 64-KiB buffer answers `NonVolatile`; the buffer keeps its own state.
        let buffer = device.new_buffer(65536, ResourceOptions::shared()).unwrap();
        let mut over = TextureDescriptor::texture_2d(rgba, 128, 128, false);
        over.usage = TextureUsage::SHADER_READ;
        let over = buffer.new_texture(&over, 0, 512).unwrap();
        assert_eq!(buffer.set_purgeable_state(Volatile), Ok(NonVolatile));
        assert_eq!(sequence(&over), never);
        assert_eq!(buffer.set_purgeable_state(NonVolatile), Ok(Volatile));

        // A 16x16 shareable texture keeps its state, one with the texture opened from its
        // handle, which starts with its label; their views answer `NonVolatile`.
        let mut private = TextureDescriptor::texture_2d(rgba, 16, 16, false);
        private.resource_options = ResourceOptions::private();
        let shared = device.new_shared_texture(&private).unwrap();
        shared.set_label("original");
        let handle = shared.new_shared_texture_handle().unwrap();
        let opened = device
            .new_shared_texture_with_handle(&handle, IdPlacement::Any)
            .unwrap();
        assert_eq!(opened.label().as_deref(), Some("original"));
        shared.set_label("relabelled");
        assert_eq!(opened.label().as_deref(), Some("original"));
        assert_eq!(opened.set_purgeable_state(Volatile), Ok(NonVolatile));
        assert_eq!(shared.set_purgeable_state(KeepCurrent), Ok(Volatile));
        let shared_view = shared.new_texture_view(rgba).unwrap();
        assert_eq!(
            shared_view.set_purgeable_state(KeepCurrent),
            Ok(NonVolatile)
        );
        assert_eq!(shared.set_purgeable_state(NonVolatile), Ok(Volatile));
        assert_eq!(opened.set_purgeable_state(KeepCurrent), Ok(NonVolatile));

        // A heap's texture answers `NonVolatile` to every request and changes nothing, its
        // heap's state included, as plain Metal answers where its validation stops it; its label is
        // its own.
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 20,
                ResourceOptions::private(),
            ))
            .unwrap();
        let in_heap = heap.new_texture(&private).unwrap();
        for state in [KeepCurrent, Volatile, Empty, NonVolatile] {
            assert_eq!(in_heap.set_purgeable_state(state), Ok(NonVolatile));
            assert_eq!(heap.set_purgeable_state(KeepCurrent), Ok(NonVolatile));
        }
        in_heap.set_label("in a heap");
        assert_eq!(in_heap.label().as_deref(), Some("in a heap"));
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}
