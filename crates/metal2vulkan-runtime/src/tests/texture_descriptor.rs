//! The texture descriptor's defaults and Metal's descriptor rules.

use super::super::error::RuntimeGap;
use super::super::texture::heap::{no_texture_alignment, texture_buffer_size_and_align};
use super::super::texture_descriptor::{
    full_chain, no_texture_alone, texture_usage, Creator, Refused, TextureDescriptor,
    TextureLayout, TextureProblem, TextureRules, TextureSwizzleChannels, APPLE_SAMPLE_COUNTS,
};
use super::super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use crate::MetalLevel;
use metal2vulkan::metal::format::ycbcr_format;
use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureType, TextureUsage};

/// A device of an Apple GPU's sample counts answering at the newest Metal level.
pub(super) const APPLE: TextureRules = TextureRules {
    sample_counts: APPLE_SAMPLE_COUNTS,
    level: MetalLevel::NEWEST,
};

/// A refusal's problem, whichever answer it has.
fn problem(refused: Refused) -> TextureProblem {
    match refused {
        Refused::Stops(problem) | Refused::Nil(problem) => problem,
        Refused::NotYet(gap) => panic!("Metal makes it; the runtime's gap {gap:?}"),
    }
}

fn of(texture_type: TextureType, extent: [u64; 3]) -> TextureDescriptor {
    TextureDescriptor {
        texture_type,
        width: extent[0],
        height: extent[1],
        depth: extent[2],
        ..TextureDescriptor::default()
    }
}

#[test]
fn the_default_is_metals() {
    let d = TextureDescriptor::default();
    assert_eq!(d.texture_type, TextureType::Type2D);
    assert_eq!(d.pixel_format, PixelFormat::RGBA8Unorm);
    assert_eq!((d.width, d.height, d.depth), (1, 1, 1));
    assert_eq!(
        (d.mipmap_level_count, d.sample_count, d.array_length),
        (1, 1, 1)
    );
    assert_eq!(d.usage, TextureUsage::SHADER_READ);
    assert_eq!(d.resource_options.raw(), 0);
    assert!(d.allow_gpu_optimized_contents);
    assert!(d.swizzle.is_default());
    let s = d.swizzle;
    assert_eq!(
        [s.red, s.green, s.blue, s.alpha],
        [
            TextureSwizzle::Red,
            TextureSwizzle::Green,
            TextureSwizzle::Blue,
            TextureSwizzle::Alpha
        ]
    );
}

#[test]
fn the_class_constructors_give_metals_descriptors() {
    let d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 640, 480, true);
    assert_eq!((d.width, d.height, d.mipmap_level_count), (640, 480, 10));
    assert_eq!(
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 640, 480, false).mipmap_level_count,
        1
    );
    let cube = TextureDescriptor::texture_cube(PixelFormat::RGBA16Float, 64, true);
    assert_eq!(cube.texture_type, TextureType::Cube);
    assert_eq!(
        (cube.width, cube.height, cube.mipmap_level_count),
        (64, 64, 7)
    );
    let buffer = TextureDescriptor::texture_buffer(
        PixelFormat::R32Float,
        1000,
        ResourceOptions::private(),
        TextureUsage::SHADER_WRITE,
    );
    assert_eq!(buffer.texture_type, TextureType::TextureBuffer);
    assert_eq!(
        (buffer.width, buffer.usage, buffer.resource_options),
        (1000, TextureUsage::SHADER_WRITE, ResourceOptions::private())
    );
}

#[test]
fn full_chains_count_to_the_largest_axis() {
    assert_eq!(full_chain(&[1, 1]), 1);
    assert_eq!(full_chain(&[2, 1]), 2);
    assert_eq!(full_chain(&[5, 5]), 3);
    assert_eq!(full_chain(&[1, 9]), 4);
    assert_eq!(full_chain(&[16, 16]), 5);
    assert_eq!(full_chain(&[16384, 1]), 15);
    assert_eq!(full_chain(&[5, 3, 7]), 3);
}

#[test]
fn a_checked_descriptor_gives_its_layout() {
    let d = TextureDescriptor {
        array_length: 3,
        mipmap_level_count: 3,
        ..of(TextureType::CubeArray, [5, 5, 1])
    };
    let layout = d.check(APPLE).unwrap();
    assert_eq!(
        layout,
        TextureLayout {
            texture_type: TextureType::CubeArray,
            pixel_format: PixelFormat::RGBA8Unorm,
            extent: [5, 5, 1],
            levels: 3,
            layers: 18,
            samples: 1,
        }
    );
    let volume = of(TextureType::Type3D, [5, 3, 7]).check(APPLE).unwrap();
    assert_eq!((volume.extent, volume.layers), ([5, 3, 7], 1));
}

#[test]
fn metals_descriptor_rules_are_problems() {
    use TextureProblem as P;
    use TextureType as T;
    let cases: Vec<(TextureDescriptor, TextureProblem)> = vec![
        (
            TextureDescriptor {
                pixel_format: PixelFormat::Invalid,
                ..of(T::Type2D, [4, 4, 1])
            },
            P::NotAFormat,
        ),
        (
            TextureDescriptor {
                pixel_format: PixelFormat::Unspecialized,
                sample_count: 4,
                ..of(T::Type2DMultisample, [4, 4, 1])
            },
            P::TypeForFormat,
        ),
        // Metal stops `ShaderAtomic` usage of `Unspecialized`, plain too.
        (
            TextureDescriptor {
                pixel_format: PixelFormat::Unspecialized,
                usage: TextureUsage::SHADER_READ
                    | TextureUsage::SHADER_WRITE
                    | TextureUsage::SHADER_ATOMIC,
                ..of(T::Type3D, [4, 4, 2])
            },
            P::UsageForFormat,
        ),
        // And of a depth or stencil format, whatever its other usage.
        (
            TextureDescriptor {
                pixel_format: PixelFormat::Depth32Float,
                usage: TextureUsage::SHADER_READ | TextureUsage::SHADER_ATOMIC,
                ..of(T::Type2D, [64, 64, 1])
            },
            P::UsageForFormat,
        ),
        (
            TextureDescriptor {
                pixel_format: PixelFormat::Stencil8,
                usage: TextureUsage::SHADER_ATOMIC,
                ..of(T::Type2D, [4, 4, 1])
            },
            P::UsageForFormat,
        ),
        (of(T::Type2D, [0, 4, 1]), P::ZeroExtent),
        (of(T::Type3D, [4, 4, 0]), P::ZeroExtent),
        (of(T::Type1D, [4, 2, 1]), P::ExtentForType),
        (of(T::Type1DArray, [4, 2, 1]), P::ExtentForType),
        (of(T::TextureBuffer, [4, 2, 1]), P::ExtentForType),
        (of(T::Type2D, [4, 4, 2]), P::ExtentForType),
        (of(T::Cube, [4, 4, 2]), P::ExtentForType),
        (of(T::Cube, [4, 5, 1]), P::CubeNotSquare),
        (of(T::CubeArray, [8, 4, 1]), P::CubeNotSquare),
        (
            TextureDescriptor {
                array_length: 2,
                ..of(T::Type2D, [4, 4, 1])
            },
            P::ArrayLength,
        ),
        (
            TextureDescriptor {
                array_length: 0,
                ..of(T::Type2DArray, [4, 4, 1])
            },
            P::ArrayLength,
        ),
        (
            TextureDescriptor {
                array_length: 2049,
                ..of(T::Type2DArray, [4, 4, 1])
            },
            P::ArrayLength,
        ),
        (
            TextureDescriptor {
                mipmap_level_count: 0,
                ..of(T::Type2D, [4, 4, 1])
            },
            P::MipmapLevelCount { max: 3 },
        ),
        (
            TextureDescriptor {
                mipmap_level_count: 4,
                ..of(T::Type2D, [4, 4, 1])
            },
            P::MipmapLevelCount { max: 3 },
        ),
        (
            TextureDescriptor {
                mipmap_level_count: 2,
                sample_count: 4,
                ..of(T::Type2DMultisample, [4, 4, 1])
            },
            P::MipmapLevelCount { max: 3 },
        ),
        (
            TextureDescriptor {
                mipmap_level_count: 4,
                ..of(T::TextureBuffer, [4, 1, 1])
            },
            P::MipmapLevelCount { max: 3 },
        ),
        // A 1D texture or array of more than one level: plain Metal asserts.
        (
            TextureDescriptor {
                mipmap_level_count: 2,
                ..of(T::Type1D, [16, 1, 1])
            },
            P::MipmapLevelCount { max: 5 },
        ),
        (
            TextureDescriptor {
                mipmap_level_count: 2,
                array_length: 2,
                ..of(T::Type1DArray, [16, 1, 1])
            },
            P::MipmapLevelCount { max: 5 },
        ),
        (
            TextureDescriptor {
                sample_count: 4,
                ..of(T::Type2D, [4, 4, 1])
            },
            P::SampleCount,
        ),
        (
            TextureDescriptor {
                sample_count: 0,
                ..of(T::Type2DMultisample, [4, 4, 1])
            },
            P::SampleCount,
        ),
        (
            of(T::TextureBuffer, [1 << 33, 1, 1]),
            P::PastLargestSide { max: 1 << 28 },
        ),
        (
            of(T::Type2D, [16385, 4, 1]),
            P::PastLargestSide { max: 16384 },
        ),
        (
            of(T::Type1D, [16385, 1, 1]),
            P::PastLargestSide { max: 16384 },
        ),
        (
            TextureDescriptor {
                array_length: 2,
                ..of(T::Type2DArray, [4, 16385, 1])
            },
            P::PastLargestSide { max: 16384 },
        ),
        (
            of(T::CubeArray, [16385, 16385, 1]),
            P::PastLargestSide { max: 16384 },
        ),
        (
            TextureDescriptor {
                sample_count: 4,
                ..of(T::Type2DMultisample, [16385, 4, 1])
            },
            P::PastLargestSide { max: 16384 },
        ),
        (
            TextureDescriptor {
                sample_count: 4,
                array_length: 2,
                ..of(T::Type2DMultisampleArray, [4, 16385, 1])
            },
            P::PastLargestSide { max: 16384 },
        ),
        (
            of(T::Type3D, [4, 4, 2049]),
            P::PastLargestSide { max: 2048 },
        ),
        // Every rule Metal stops comes before `Unspecialized`, which it answers nil for.
        (
            TextureDescriptor {
                pixel_format: PixelFormat::Unspecialized,
                ..of(T::Type2D, [0, 4, 1])
            },
            P::ZeroExtent,
        ),
        (
            TextureDescriptor {
                swizzle: gbar(),
                usage: TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
                ..of(T::Type2D, [4, 4, 1])
            },
            P::SwizzleWithWrites,
        ),
        (
            TextureDescriptor {
                swizzle: gbar(),
                usage: TextureUsage::RENDER_TARGET,
                ..of(T::Type2D, [4, 4, 1])
            },
            P::SwizzleWithWrites,
        ),
    ];
    for (desc, want) in cases {
        assert_eq!(desc.check(APPLE), Err(Refused::Stops(want)), "{desc:?}");
    }
    // Plain Metal makes no `Unspecialized` texture and answers nil.
    let unspecialized = TextureDescriptor {
        pixel_format: PixelFormat::Unspecialized,
        ..of(T::Type2D, [4, 4, 1])
    };
    assert_eq!(
        unspecialized.check(APPLE),
        Err(Refused::Nil(P::Unspecialized))
    );
    // A texture buffer's width reaches past 16384, to 2^28.
    assert!(of(T::TextureBuffer, [16385, 1, 1]).check(APPLE).is_ok());
    assert!(of(T::TextureBuffer, [1 << 28, 1, 1]).check(APPLE).is_ok());
    assert_eq!(
        of(T::TextureBuffer, [(1 << 28) + 1, 1, 1]).check(APPLE),
        Err(Refused::Stops(P::PastLargestSide { max: 1 << 28 }))
    );
}

/// A multisample texture's sample count is above 1 and one the device takes: on an Apple GPU 2 or
/// 4, where Metal stops 1, 3, 8 and 16, and 8 on a device that takes it. The shared creator answers
/// nil first for `RGBA16Float_XR`, of which Metal makes no texture alone, where the device's
/// creator stops its unsupported count.
#[test]
fn a_multisample_texture_takes_a_count_the_device_takes() {
    use TextureProblem as P;
    use TextureType as T;
    let ms = |pf: PixelFormat, samples: u64| TextureDescriptor {
        pixel_format: pf,
        sample_count: samples,
        usage: TextureUsage::RENDER_TARGET,
        ..of(T::Type2DMultisample, [16, 16, 1])
    };
    for pf in [PixelFormat::RGBA8Unorm, PixelFormat::RGBA16Float] {
        for samples in [2, 4] {
            assert!(ms(pf, samples).check(APPLE).is_ok(), "{pf:?} {samples}");
            assert!(
                ms(pf, samples).check_shared(APPLE).is_ok(),
                "{pf:?} {samples}"
            );
        }
        for samples in [1, 3, 8, 16] {
            for checked in [
                ms(pf, samples).check(APPLE),
                ms(pf, samples).check_shared(APPLE),
            ] {
                assert_eq!(
                    checked,
                    Err(Refused::Stops(P::SampleCount)),
                    "{pf:?} {samples}"
                );
            }
        }
    }
    let eight = TextureRules {
        sample_counts: metal2vulkan::vk::SampleCountFlags::from_raw(0b1111),
        ..APPLE
    };
    assert!(ms(PixelFormat::RGBA8Unorm, 8).check(eight).is_ok());
    // A count an Apple GPU takes and the device does not (1 and 4 alone) is the device's gap,
    // refused as the image is planned, not a stop; one neither takes is stopped.
    let one_and_four = TextureRules {
        sample_counts: metal2vulkan::vk::SampleCountFlags::from_raw(0b101),
        ..APPLE
    };
    assert!(ms(PixelFormat::RGBA8Unorm, 2).check(one_and_four).is_ok());
    assert_eq!(
        ms(PixelFormat::RGBA8Unorm, 8).check(one_and_four),
        Err(Refused::Stops(P::SampleCount))
    );
    let xr = PixelFormat::RGBA16FloatXR;
    assert_eq!(ms(xr, 8).check(APPLE), Err(Refused::Stops(P::SampleCount)));
    for samples in [4, 8] {
        assert_eq!(
            ms(xr, samples).check_shared(APPLE),
            Err(Refused::Nil(P::NoTextureOfFormat)),
            "{samples}"
        );
    }
    assert_eq!(
        ms(xr, 4).check(APPLE),
        Err(Refused::Nil(P::NoTextureOfFormat))
    );
}

/// A texture buffer takes its width's full chain of levels, which Metal's device, heap and shared
/// creators make and the runtime does not build yet; a buffer's creator stops more than one
/// (`linear_plan`), and the heap size query answers the first level's size (256 bytes at 16 for 64
/// `RGBA8Unorm` texels of 2 levels).
#[test]
fn a_texture_buffer_of_levels_is_the_runtimes_gap() {
    use TextureProblem as P;
    let counts = APPLE;
    let levels = |count: u64, height: u64| TextureDescriptor {
        mipmap_level_count: count,
        height,
        ..TextureDescriptor::texture_buffer(
            PixelFormat::RGBA8Unorm,
            64,
            ResourceOptions::private(),
            TextureUsage::SHADER_READ,
        )
    };
    let gap = Err(Refused::NotYet(RuntimeGap::TextureBufferLevels));
    for count in [2, 7] {
        let d = levels(count, 1);
        assert_eq!(d.check(counts), gap, "{count}");
        assert_eq!(d.check_shared(counts), gap, "{count}");
        let linear = d.check_made(Creator::Buffer, counts).unwrap();
        assert_eq!(linear.levels as u64, count);
        let sized = d.check_made(Creator::SizeQuery, counts).unwrap();
        let answer = texture_buffer_size_and_align(&sized).unwrap();
        assert_eq!((answer.size, answer.align), (256, 16), "{count}");
    }
    for creator in [
        Creator::Device,
        Creator::Buffer,
        Creator::Shared,
        Creator::SizeQuery,
    ] {
        assert_eq!(
            levels(8, 1).check_made(creator, counts),
            Err(Refused::Stops(P::MipmapLevelCount { max: 7 })),
            "{creator:?}"
        );
        assert_eq!(
            levels(0, 1).check_made(creator, counts),
            Err(Refused::Stops(P::MipmapLevelCount { max: 7 })),
            "{creator:?}"
        );
        assert_eq!(
            levels(2, 2).check_made(creator, counts),
            Err(Refused::Stops(P::ExtentForType)),
            "{creator:?}"
        );
    }
    assert!(levels(1, 1).check(counts).is_ok());
}

/// Under `write_swizzle_enabled` a swizzle takes writes and render-target usage, which Metal
/// makes from the device's, a heap's and a buffer's creators and the runtime does not build yet;
/// the shared creator still stops it, and the heap size query answers for it, sized with the
/// usage Metal's texture reports.
#[test]
fn a_write_swizzle_is_the_runtimes_gap() {
    let counts = APPLE;
    let bgra = TextureSwizzleChannels::new(
        TextureSwizzle::Blue,
        TextureSwizzle::Green,
        TextureSwizzle::Red,
        TextureSwizzle::Alpha,
    );
    let read_write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let rendered = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let of_usage = |usage: TextureUsage, swizzle, enabled| TextureDescriptor {
        usage,
        swizzle,
        write_swizzle_enabled: enabled,
        ..TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false)
    };
    let gap = Err(Refused::NotYet(RuntimeGap::WriteSwizzle));
    let stop = Err(Refused::Stops(TextureProblem::SwizzleWithWrites));
    for usage in [
        read_write,
        rendered,
        read_write | TextureUsage::RENDER_TARGET,
    ] {
        let d = of_usage(usage, bgra, true);
        assert_eq!(d.check(counts), gap, "{usage:?}");
        assert_eq!(d.check_made(Creator::Buffer, counts), gap, "{usage:?}");
        assert_eq!(d.check_over_surface(counts), gap, "{usage:?}");
        assert_eq!(d.check_shared(counts), stop, "{usage:?}");
        assert!(d.check_made(Creator::SizeQuery, counts).is_ok());
        assert_eq!(d.image_usage(), usage, "{usage:?}");
        let off = of_usage(usage, bgra, false);
        for creator in [
            Creator::Device,
            Creator::Buffer,
            Creator::Shared,
            Creator::SizeQuery,
        ] {
            assert_eq!(
                off.check_made(creator, counts),
                stop,
                "{usage:?} {creator:?}"
            );
        }
    }
    // Without a swizzle, or without writes, it changes nothing.
    let plain = of_usage(read_write, TextureSwizzleChannels::DEFAULT, true);
    assert!(plain.check(counts).is_ok());
    let read = of_usage(TextureUsage::SHADER_READ, bgra, true);
    assert!(read.check(counts).is_ok());
    assert_eq!(read.image_usage(), TextureUsage::SHADER_READ);
}

fn gbar() -> TextureSwizzleChannels {
    TextureSwizzleChannels::new(
        TextureSwizzle::Green,
        TextureSwizzle::Blue,
        TextureSwizzle::Alpha,
        TextureSwizzle::Red,
    )
}

#[test]
fn swizzled_textures_are_read_only_and_identity_swizzles_are_not_swizzles() {
    let read = TextureDescriptor {
        swizzle: gbar(),
        usage: TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        ..TextureDescriptor::default()
    };
    assert!(read.check(APPLE).is_ok());
    let identity = TextureDescriptor {
        swizzle: TextureSwizzleChannels::new(
            TextureSwizzle::Red,
            TextureSwizzle::Green,
            TextureSwizzle::Blue,
            TextureSwizzle::Alpha,
        ),
        usage: TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET,
        ..TextureDescriptor::default()
    };
    assert!(identity.check(APPLE).is_ok());
    // Probe 4 (`swz.desc.unknown`): a swizzled descriptor of `Unknown` usage is made, reporting
    // `ShaderRead | PixelFormatView`; without a swizzle `Unknown` reports 0x17.
    let unknown = TextureDescriptor {
        swizzle: gbar(),
        usage: TextureUsage::empty(),
        ..TextureDescriptor::default()
    };
    assert!(unknown.check(APPLE).is_ok());
    assert_eq!(texture_usage(unknown.usage, unknown.swizzle).as_raw(), 0x11);
    assert_eq!(
        texture_usage(TextureUsage::empty(), TextureSwizzleChannels::DEFAULT).as_raw(),
        0x17
    );
    assert_eq!(
        texture_usage(TextureUsage::SHADER_READ, gbar()),
        TextureUsage::SHADER_READ
    );
}

#[test]
fn array_lengths_and_samples_metal_takes_are_accepted() {
    let arrays = TextureDescriptor {
        array_length: 2048,
        ..of(TextureType::Type2DArray, [4, 4, 1])
    };
    assert_eq!(arrays.check(APPLE).unwrap().layers, 2048);
    let one = TextureDescriptor {
        array_length: 1,
        ..of(TextureType::Type1DArray, [4, 1, 1])
    };
    assert_eq!(one.check(APPLE).unwrap().layers, 1);
    let ms = TextureDescriptor {
        sample_count: 4,
        array_length: 2,
        ..of(TextureType::Type2DMultisampleArray, [4, 4, 1])
    };
    let layout = ms.check(APPLE).unwrap();
    assert_eq!((layout.samples, layout.layers, layout.levels), (4, 2, 1));
}

#[test]
fn swizzle_channels_name_their_sources() {
    let s = gbar();
    assert!(!s.is_default());
    assert_eq!(s.alpha, TextureSwizzle::Red);
    assert_eq!(
        TextureSwizzleChannels::default(),
        TextureSwizzleChannels::DEFAULT
    );
}

/// Probe 5's verdicts (Metal's API validation on an Apple M3 Max, macOS 26.7, and an M4, macOS
/// 27.0.1, the same on both): the texture types each pixel format takes, of 1D, 1D array, 2D,
/// 2D array, cube, cube array and 3D; `X32_Stencil8` takes plain Metal's, a stencil format's.
#[test]
fn texture_types_follow_probe_5s_verdicts_per_format() {
    use TextureType as T;
    const TYPES: [TextureType; 7] = [
        T::Type1D,
        T::Type1DArray,
        T::Type2D,
        T::Type2DArray,
        T::Cube,
        T::CubeArray,
        T::Type3D,
    ];
    let row = |s: &str| -> [bool; 7] {
        let mut out = [false; 7];
        for (o, c) in out.iter_mut().zip(s.chars()) {
            *o = c == 'o';
        }
        out
    };
    let rows: [(&str, &[&str]); 6] = [
        // ASTC's 4x2 and 8x4 formats too, measured beside `ASTC_4x4_LDR`.
        ("aaooooo", &["ASTC_", "BC"]),
        // Metal answers nil for a cube and a cube array.
        ("ooooaao", &["RGB10A8_2P_XR10", "RGB10A8_2P_XR10_sRGB"]),
        ("aaoaaaa", &["GBGR422", "BGRG422"]),
        (
            "aaooooa",
            &[
                "Depth16Unorm",
                "Depth32Float",
                "Depth32Float_Stencil8",
                "Stencil8",
                // A texture made in it is a stencil texture.
                "X32_Stencil8",
                "EAC_",
                "ETC2_",
                "PVRTC_",
                // Apple silicon has no such format at all ("invalid pixelFormat" in every
                // type); Metal's type rule for depth formats is the one stated.
                "Depth24Unorm_Stencil8",
            ],
        ),
        ("aaaaaaa", &["X24_Stencil8"]),
        ("ooooooo", &[""]),
    ];
    for &pf in PixelFormat::ALL {
        if matches!(pf, PixelFormat::Invalid | PixelFormat::Unspecialized) {
            continue;
        }
        // Probe 5 took no format Metal makes no texture of from a descriptor alone, which
        // answers nil for every type here, but `AnyDepth` and `AnyStencil`, whose 1D, 1D array
        // and 3D textures Metal stops.
        if no_texture_alone(pf) {
            let any_depth = matches!(pf, PixelFormat::AnyDepth | PixelFormat::AnyStencil);
            for ty in TYPES {
                let one_dimensional = matches!(ty, T::Type1D | T::Type1DArray);
                let mut d = of(ty, [8, if one_dimensional { 1 } else { 8 }, 1]);
                d.pixel_format = pf;
                let want = if any_depth && (one_dimensional || ty == T::Type3D) {
                    Refused::Stops(TextureProblem::TypeForFormat)
                } else {
                    Refused::Nil(TextureProblem::NoTextureOfFormat)
                };
                assert_eq!(d.check(APPLE).map(|_| ()), Err(want), "{pf:?} {ty:?}");
            }
            continue;
        }
        // Nor one the level describes none of (24-bit depth among them), stopped in every type
        // (`tests::metal_level`).
        if !APPLE.level.describes(pf) {
            continue;
        }
        let name = &pf.name()["MTLPixelFormat".len()..];
        let (pattern, _) = rows
            .iter()
            .find(|(_, prefixes)| {
                prefixes.iter().any(|p| {
                    if p.ends_with('_') || *p == "BC" {
                        name.starts_with(p)
                    } else {
                        p.is_empty() || name == *p
                    }
                })
            })
            .unwrap();
        for (ty, ok) in TYPES.iter().zip(row(pattern)) {
            let mut d = of(
                *ty,
                [
                    8,
                    if matches!(ty, T::Type1D | T::Type1DArray) {
                        1
                    } else {
                        8
                    },
                    1,
                ],
            );
            d.pixel_format = pf;
            if *ty == T::Type3D {
                d.depth = 4;
            }
            if matches!(ty, T::Type1DArray | T::Type2DArray | T::CubeArray) {
                d.array_length = 2;
            }
            let got = d.check(APPLE).map(|_| ()).map_err(problem);
            if ok {
                assert_eq!(got, Ok(()), "{name} {ty:?}");
            } else {
                assert!(
                    matches!(
                        got,
                        Err(TextureProblem::TypeForFormat | TextureProblem::ViewOnlyFormat)
                    ),
                    "{name} {ty:?}: {got:?}"
                );
            }
        }
    }
}

/// A cube array of more than 341 cubes is stopped for every format, before a nil (`RGBA8Unorm`,
/// `BGRA10_XR`, the two-plane extended-range and YCbCr formats, plain Metal too); a memoryless
/// two-plane extended-range cube passes the descriptor's rules, which Metal makes, and one of
/// another storage mode is nil.
#[test]
fn cube_arrays_stop_past_341_cubes_and_memoryless_two_plane_cubes_are_not_nil() {
    use TextureType as T;
    for pf in [
        PixelFormat::RGBA8Unorm,
        PixelFormat::BGRA10XR,
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RGB10A82PXR10SRGB,
        PixelFormat::YCBCR84202P,
    ] {
        for (length, stops) in [(341, false), (342, true), (2048, true)] {
            let mut d = of(T::CubeArray, [16, 16, 1]);
            d.pixel_format = pf;
            d.array_length = length;
            let checked = d.check(APPLE);
            assert_eq!(
                matches!(checked, Err(Refused::Stops(TextureProblem::ArrayLength))),
                stops,
                "{pf:?} {length}: {:?}",
                checked.map(|layout| layout.layers)
            );
        }
    }
    for pf in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
        for (ty, length) in [(T::Cube, 1), (T::CubeArray, 341)] {
            let mut d = of(ty, [16, 16, 1]);
            d.pixel_format = pf;
            d.array_length = length;
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            assert_eq!(
                d.check(APPLE).map(|_| ()),
                Err(Refused::Nil(TextureProblem::TypeForFormat)),
                "{pf:?} {ty:?}"
            );
            d.resource_options = ResourceOptions::new(
                StorageMode::Memoryless,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            assert_eq!(
                d.check(APPLE).map(|layout| layout.layers),
                Ok(6 * length as u32)
            );
        }
    }
}

/// The padded 10-bit formats (`R10Unorm_X6`, `RG10Unorm_X12` and their `_sRGB` twins) take no
/// multisample texture, which Metal's validation stops, of either multisample type; a 2D one
/// passes the descriptor's rules.
#[test]
fn padded_formats_take_no_multisample_texture() {
    use TextureType as T;
    for pf in [
        PixelFormat::R10UnormX6,
        PixelFormat::R10UnormX6SRGB,
        PixelFormat::RG10UnormX12,
        PixelFormat::RG10UnormX12SRGB,
    ] {
        for (ty, slices) in [(T::Type2DMultisample, 1), (T::Type2DMultisampleArray, 2)] {
            let mut d = of(ty, [16, 16, 1]);
            d.pixel_format = pf;
            d.sample_count = 4;
            d.array_length = slices;
            d.usage = TextureUsage::RENDER_TARGET;
            assert_eq!(
                d.check(APPLE).map(|_| ()),
                Err(Refused::Stops(TextureProblem::TypeForFormat)),
                "{pf:?} {ty:?}"
            );
        }
        let mut d = of(T::Type2D, [16, 16, 1]);
        d.pixel_format = pf;
        assert!(d.check(APPLE).is_ok(), "{pf:?}");
    }
}

/// The two-plane extended-range formats take Metal's measured rules (an
/// Apple M4 on macOS 27.0.1): a multisample texture and `ShaderAtomic` usage are stopped (Metal's
/// validation asserts); 2D textures of every other usage, mipmapped, and 2D array, 1D and 3D ones
/// are made.
#[test]
fn two_plane_extended_range_textures_follow_metal_s_rules() {
    use TextureType as T;
    for pf in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
        let mut d = of(T::Type2DMultisample, [64, 4, 1]);
        d.pixel_format = pf;
        d.sample_count = 2;
        d.usage = TextureUsage::RENDER_TARGET;
        assert_eq!(
            d.check(APPLE).map(|_| ()),
            Err(Refused::Stops(TextureProblem::TypeForFormat)),
            "{pf:?}"
        );
        // Metal answers nil for a cube and a cube array.
        for (ty, slices) in [(T::Cube, 1), (T::CubeArray, 2)] {
            let mut d = of(ty, [32, 32, 1]);
            d.pixel_format = pf;
            d.array_length = slices;
            assert_eq!(
                d.check(APPLE).map(|_| ()),
                Err(Refused::Nil(TextureProblem::TypeForFormat)),
                "{pf:?} {ty:?}"
            );
        }
        let mut d = of(T::Type2D, [64, 4, 1]);
        d.pixel_format = pf;
        d.usage =
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC;
        assert_eq!(
            d.check(APPLE).map(|_| ()),
            Err(Refused::Stops(TextureProblem::UsageForFormat)),
            "{pf:?}"
        );
        for usage in [
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_WRITE,
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            TextureUsage::RENDER_TARGET,
            TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        ] {
            let mut d = of(T::Type2D, [63, 5, 1]);
            d.pixel_format = pf;
            d.usage = usage;
            d.mipmap_level_count = 2;
            assert!(d.check(APPLE).is_ok(), "{pf:?} {usage:?}");
        }
    }
}

/// Probe 6's verdicts (Metal on an Apple M3 Max, macOS 26.7, and an M4, macOS 27.0.1): a PVRTC
/// texture is square with a power-of-two side, a 4:2:2 one has an even width, and the other
/// formats take every extent measured.
#[test]
fn extents_follow_probe_6s_verdicts_per_format() {
    use TextureType as T;
    const FLAT: [[u64; 2]; 12] = [
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 5],
        [8, 8],
        [8, 4],
        [4, 8],
        [12, 12],
        [16, 8],
    ];
    for &pf in PixelFormat::ALL {
        if matches!(pf, PixelFormat::Invalid | PixelFormat::Unspecialized) {
            continue;
        }
        // Probe 6 took no format Metal makes no texture of from a descriptor alone.
        if no_texture_alone(pf) {
            continue;
        }
        // Nor one the level describes none of (24-bit depth among them), stopped before every
        // extent rule (`tests::metal_level`).
        if !APPLE.level.describes(pf) {
            continue;
        }
        let name = &pf.name()["MTLPixelFormat".len()..];
        let pvrtc = name.starts_with("PVRTC_");
        let four_two_two = matches!(pf, PixelFormat::GBGR422 | PixelFormat::BGRG422);
        for (ty, extents) in [
            (T::Type2D, &FLAT[..]),
            (T::Type2DArray, &[[8, 8], [6, 6], [7, 5]][..]),
            (T::Cube, &[[8, 8], [6, 6]][..]),
            (T::CubeArray, &[[8, 8], [6, 6]][..]),
        ] {
            for &[w, h] in extents {
                let mut d = of(ty, [w, h, 1]);
                d.pixel_format = pf;
                if matches!(ty, T::Type2DArray | T::CubeArray) {
                    d.array_length = 2;
                }
                let got = d.check(APPLE).map(|_| ()).map_err(problem);
                let refused_by_type = matches!(
                    got,
                    Err(TextureProblem::TypeForFormat | TextureProblem::ViewOnlyFormat)
                );
                if refused_by_type {
                    continue;
                }
                let refused = (pvrtc && (w != h || !w.is_power_of_two()))
                    || (four_two_two && !w.is_multiple_of(2));
                if refused {
                    assert_eq!(
                        got,
                        Err(TextureProblem::ExtentForFormat),
                        "{name} {ty:?} {w}x{h}"
                    );
                } else {
                    assert_eq!(got, Ok(()), "{name} {ty:?} {w}x{h}");
                }
            }
        }
        // 3D: the types probe 5 refuses stay refused for their type; the rest take each extent.
        for extent in [[4, 4, 4], [7, 5, 3], [8, 8, 2]] {
            let mut d = of(T::Type3D, extent);
            d.pixel_format = pf;
            let got = d.check(APPLE).map(|_| ()).map_err(problem);
            assert!(
                matches!(
                    got,
                    Ok(()) | Err(TextureProblem::TypeForFormat | TextureProblem::ViewOnlyFormat)
                ),
                "{name} 3D {extent:?}: {got:?}"
            );
        }
    }
}

/// Metal refuses a mipmapped 4:2:2 texture, plain too (16 x 16, 18 x 6 and 6 x
/// 6 of their full chains, "the MTLPixelFormat422 pixel format does not support mipmaps"); one
/// level is made.
#[test]
fn a_four_two_two_texture_has_one_level() {
    for pf in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
        for [w, h] in [[16, 16], [18, 6], [6, 6]] {
            let mut d = TextureDescriptor::texture_2d(pf, w, h, true);
            assert!(
                matches!(
                    d.check(APPLE),
                    Err(Refused::Stops(TextureProblem::MipmapLevelCount { .. }))
                ),
                "{pf:?} {w}x{h}: {:?}",
                d.check(APPLE)
            );
            d.mipmap_level_count = 1;
            assert_eq!(d.check(APPLE).map(|l| l.levels), Ok(1), "{pf:?} {w}x{h}");
        }
    }
}

/// A rotation is 0 to 4 and needs `framebuffer_only`, as Metal's descriptor validation asserts,
/// plain too: past 4 is `Rotation` whatever else, a nonzero one without the flag
/// `RotationWithoutFramebufferOnly`, and with it every rotation up to 4 checks as the texture it
/// rotates.
#[test]
fn a_rotation_is_up_to_four_and_framebuffer_only() {
    let rotated = |rotation: u64, framebuffer_only: bool| TextureDescriptor {
        rotation,
        framebuffer_only,
        usage: TextureUsage::RENDER_TARGET,
        ..TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 64, 16, false)
    };
    let plain = rotated(0, false).check(APPLE).unwrap();
    for rotation in 0..=4 {
        assert_eq!(
            rotated(rotation, true).check(APPLE),
            Ok(plain),
            "{rotation}"
        );
    }
    for rotation in 1..=4 {
        assert_eq!(
            rotated(rotation, false).check(APPLE),
            Err(Refused::Stops(
                TextureProblem::RotationWithoutFramebufferOnly
            )),
            "{rotation}"
        );
    }
    for rotation in [5, 7, u64::MAX] {
        for framebuffer_only in [false, true] {
            assert_eq!(
                rotated(rotation, framebuffer_only).check(APPLE),
                Err(Refused::Stops(TextureProblem::Rotation)),
                "{rotation}"
            );
        }
    }
    let d = TextureDescriptor::default();
    assert_eq!(
        (d.rotation, d.framebuffer_only, d.is_drawable),
        (0, false, false)
    );
}

/// The two-plane YCbCr formats take Metal's measured rules (an Apple M4 on macOS 27.0.1, each of
/// the 69): a 2D multisample texture is stopped, over a surface too, and a 2D multisample array;
/// plain creation answers nil for 1D, 1D array, 2D, 2D array, 3D, cube, cube array and texture
/// buffer textures, whose heap query answers 0 bytes at 16, or 128 for the 2D array, once every
/// rule Metal stops passed; and over a surface `YCBCR8_420_2P`, `YCBCR10_420_2P` and `RGB10_420_2P`
/// are made, the others refused until their reads are exact.
#[test]
fn two_plane_ycbcr_textures_follow_metal_s_rules() {
    use TextureType as T;
    let family: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| ycbcr_format(pf).is_some())
        .collect();
    assert_eq!(family.len(), 69);
    for pf in family {
        let mut d = of(T::Type2DMultisample, [64, 64, 1]);
        d.pixel_format = pf;
        d.sample_count = 4;
        for made in [d.check(APPLE), d.check_over_surface(APPLE)] {
            assert_eq!(
                made.map(|_| ()),
                Err(Refused::Stops(TextureProblem::TypeForFormat)),
                "{pf:?}"
            );
        }
        let mut d = of(T::Type2DMultisampleArray, [64, 64, 1]);
        d.pixel_format = pf;
        d.sample_count = 4;
        d.array_length = 2;
        assert_eq!(
            d.check(APPLE).map(|_| ()),
            Err(Refused::Stops(TextureProblem::TypeForFormat)),
            "{pf:?}"
        );
        for (ty, extent, slices, align) in [
            (T::Type1D, [64, 1, 1], 1, 16),
            (T::Type1DArray, [64, 1, 1], 2, 16),
            (T::Type2D, [64, 64, 1], 1, 16),
            (T::Type2DArray, [64, 64, 1], 2, 128),
            (T::Type3D, [64, 64, 4], 1, 16),
            (T::Cube, [64, 64, 1], 1, 16),
            (T::CubeArray, [64, 64, 1], 2, 16),
            (T::TextureBuffer, [64, 1, 1], 1, 16),
        ] {
            let mut d = of(ty, extent);
            d.pixel_format = pf;
            d.array_length = slices;
            assert_eq!(
                d.check(APPLE).map(|_| ()),
                Err(Refused::Nil(TextureProblem::NoTextureOfFormat)),
                "{pf:?} {ty:?}"
            );
            for level in MetalLevel::ALL {
                assert_eq!(
                    no_texture_alignment(pf, ty, level),
                    align,
                    "{pf:?} {ty:?} {level}"
                );
            }
            // Each rule Metal stops comes first (Y3): an extent of 0 here.
            d.width = 0;
            assert_eq!(
                d.check(APPLE).map(|_| ()),
                Err(Refused::Stops(TextureProblem::ZeroExtent)),
                "{pf:?} {ty:?}"
            );
        }
        let mut d = of(T::Type2D, [64, 64, 1]);
        d.pixel_format = pf;
        assert_eq!(
            d.check_over_surface(APPLE).is_ok(),
            matches!(
                pf,
                PixelFormat::YCBCR84202P
                    | PixelFormat::YCBCR104202P
                    | PixelFormat::RGB84202P
                    | PixelFormat::RGB104202P
                    | PixelFormat::YCBCR84202PSRGB
                    | PixelFormat::YCBCR104202PSRGB
                    | PixelFormat::YCBCR84222P
                    | PixelFormat::YCBCR84442P
                    | PixelFormat::YCBCR104222P
                    | PixelFormat::YCBCR104442P
                    | PixelFormat::YCBCR84222PSRGB
                    | PixelFormat::YCBCR84442PSRGB
                    | PixelFormat::YCBCR104222PSRGB
                    | PixelFormat::YCBCR104442PSRGB
                    | PixelFormat::RGB84222P
                    | PixelFormat::RGB84442P
                    | PixelFormat::RGB104222P
                    | PixelFormat::RGB104442P
                    | PixelFormat::YCBCR104202PPACKED
                    | PixelFormat::YCBCR104222PPACKED
                    | PixelFormat::YCBCR104442PPACKED
                    | PixelFormat::YCBCR104202PPACKEDSRGB
                    | PixelFormat::YCBCR104222PPACKEDSRGB
                    | PixelFormat::YCBCR104442PPACKEDSRGB
            ),
            "{pf:?}"
        );
    }
}
